"""Capture a public Binance BTC options chain and calibrate a diagnostic IV surface.

Usage: python -m volterra.options_market --out data/binance_options_snapshot
This is a point-in-time market snapshot, not historical options-chain data.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from .analytics import implied_volatility

BASE = "https://eapi.binance.com/eapi/v1"
YEAR_MS = 365.0 * 24 * 60 * 60 * 1000


def _get(endpoint: str, params: dict | None = None) -> tuple[object, int, float]:
    query = "?" + urlencode(params) if params else ""
    request = Request(f"{BASE}/{endpoint}{query}", headers={"User-Agent": "volterra-research/0.1"})
    started = time.time()
    for attempt in range(5):
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
            return payload, int(time.time() * 1000), time.time() - started
        except HTTPError as exc:
            if exc.code not in (418, 429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f"Binance {endpoint} HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
            delay = float(exc.headers.get("Retry-After", 1))
            time.sleep(min(delay, 30.0) * (attempt+1))
        except (URLError, TimeoutError) as exc:
            if attempt == 4:
                raise RuntimeError(f"Binance {endpoint} request failed: {exc}") from exc
            time.sleep(.5*(2**attempt))
    raise RuntimeError(f"Binance {endpoint} request failed after retries")


def fetch_btc_chain() -> tuple[dict, list[dict]]:
    # Read-only public market data. Parallel calls reduce cross-endpoint skew.
    names = ("exchangeInfo", "mark", "ticker", "index")
    params = {"index": {"underlying": "BTCUSDT"}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda name: _get(name, params.get(name)), names))
    payloads = dict(zip(names, (result[0] for result in results)))
    captured = dict(zip(names, (result[1] for result in results)))
    latencies = dict(zip(names, (result[2] for result in results)))
    exchange = payloads["exchangeInfo"]
    index = payloads["index"]
    asof = int(index["time"])
    marks = {row["symbol"]: row for row in payloads["mark"]}
    tickers = {row["symbol"]: row for row in payloads["ticker"]}
    contracts = [row for row in exchange.get("optionSymbols", [])
                 if row.get("underlying") == "BTCUSDT" and row.get("status") == "TRADING"
                 and int(row["expiryDate"]) > asof]
    raw = []
    for c in contracts:
        symbol = c["symbol"]
        ticker = tickers.get(symbol, {})
        mark = marks.get(symbol, {})
        if not ticker or not mark:
            continue
        raw.append({
            "symbol": symbol, "side": c["side"].lower(), "strike": float(c["strikePrice"]),
            "expiry_ms": int(c["expiryDate"]), "unit": float(c.get("unit", 1)),
            "status": c["status"], "bid": _number(ticker.get("bidPrice")),
            "ask": _number(ticker.get("askPrice")), "last": _number(ticker.get("lastPrice")),
            "volume_24h": _number(ticker.get("volume")), "trade_count_24h": int(ticker.get("tradeCount", 0)),
            "last_trade_ms": int(ticker.get("closeTime", 0)),
            "mark_price": _number(mark.get("markPrice")), "mark_iv": _number(mark.get("markIV")),
            "bid_iv_exchange": _number(mark.get("bidIV")), "ask_iv_exchange": _number(mark.get("askIV")),
            "risk_free_rate": _number(mark.get("riskFreeInterest")),
        })
    context = {
        "provider": "Binance Options public REST API", "underlying": "BTCUSDT",
        "captured_at_utc": datetime.fromtimestamp(asof/1000, timezone.utc).isoformat(),
        "index_time_ms": asof, "spot_index": float(index["indexPrice"]),
        "exchange_server_time_ms": int(exchange["serverTime"]),
        "endpoint_capture_time_ms": captured, "endpoint_latency_seconds": latencies,
        "trading_symbols_in_exchange_info": len(contracts), "joined_contract_rows": len(raw),
        "market_data_endpoints": {name: f"{BASE}/{name}" for name in names},
    }
    return context, raw


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def enrich_quotes(context: dict, raw: list[dict]) -> list[dict]:
    s = context["spot_index"]
    asof = context["index_time_ms"]
    if not math.isfinite(s) or s <= 0:
        raise ValueError("Binance returned an invalid BTCUSDT index price")
    usable = []
    for row in raw:
        t = (row["expiry_ms"] - asof) / YEAR_MS
        row["time_years"] = t
        row["forward"] = s * math.exp(row["risk_free_rate"] * t)
        row["log_forward_moneyness"] = math.log(row["strike"] / row["forward"])
        row["mid"] = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 and row["ask"] >= row["bid"] else math.nan
        row["relative_spread"] = ((row["ask"]-row["bid"])/row["mid"]
                                   if math.isfinite(row["mid"]) and row["mid"] > 0 else math.nan)
        row["quote_status"] = "eligible"
        if not math.isfinite(row["risk_free_rate"]):
            row["quote_status"] = "missing_risk_free_rate"
        elif not (row["bid"] > 0 and row["ask"] >= row["bid"] and math.isfinite(row["mid"])):
            row["quote_status"] = "missing_or_crossed_two_sided_quote"
        elif t <= 0:
            row["quote_status"] = "expired_or_nonpositive_maturity"
        else:
            usable.append(row)
    if usable:
        types = [row["side"] for row in usable]
        market = np.asarray([row["mid"] for row in usable])
        strike = np.asarray([row["strike"] for row in usable])
        times = np.asarray([row["time_years"] for row in usable])
        rates = np.asarray([row["risk_free_rate"] for row in usable])
        for side in ("call", "put"):
            mask = np.asarray([value == side for value in types])
            vols = implied_volatility(market[mask], np.full(mask.sum(), s), strike[mask], times[mask], rates[mask], option=side)
            indices = np.flatnonzero(mask)
            for idx, vol in zip(indices, np.atleast_1d(vols)):
                if not np.isfinite(vol) or vol <= 0 or vol > 5:
                    usable[idx]["quote_status"] = "mid_price_no_finite_iv"
                    continue
                usable[idx]["mid_iv"] = float(vol)
                usable[idx]["iv_vs_exchange_mark"] = float(vol)-usable[idx]["mark_iv"] if math.isfinite(usable[idx]["mark_iv"]) else math.nan
                usable[idx]["trade_age_seconds"] = max(0.0, (asof-usable[idx]["last_trade_ms"])/1000) if usable[idx]["last_trade_ms"] else math.nan
    return raw


def _fit_smile(rows: list[dict]) -> dict | None:
    # Fit IV as a quadratic in scaled log-forward-moneyness, weighted by quote
    # uncertainty (bid/ask IV width where valid, otherwise market-price spread).
    good = [r for r in rows if r.get("quote_status") == "eligible" and math.isfinite(r.get("mid_iv", math.nan))]
    if len({r["strike"] for r in good}) < 5:
        return None
    x = np.asarray([r["log_forward_moneyness"] for r in good])
    y = np.asarray([r["mid_iv"] for r in good])
    scale = float(max(np.max(np.abs(x)), 1e-4))
    xs = x/scale
    bid_iv = np.asarray([r["bid_iv_exchange"] for r in good])
    ask_iv = np.asarray([r["ask_iv_exchange"] for r in good])
    iv_width = np.where(np.isfinite(bid_iv) & np.isfinite(ask_iv) & (ask_iv >= bid_iv), ask_iv-bid_iv,
                        np.asarray([max(r["relative_spread"], .05) for r in good]))
    weights = 1.0/np.maximum(iv_width, .005)**2
    weights = np.minimum(weights, np.quantile(weights, .9))
    design = np.column_stack((np.ones(len(xs)), xs, xs*xs))
    root_w = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design*root_w[:, None], y*root_w, rcond=None)
    predicted = design@beta
    residual = predicted-y
    rmse = float(np.sqrt(np.average(residual**2, weights=weights)))
    return {"expiry_ms": good[0]["expiry_ms"], "expiry_date_utc": datetime.fromtimestamp(good[0]["expiry_ms"]/1000, timezone.utc).date().isoformat(),
            "time_years": good[0]["time_years"], "x_scale": scale, "coefficients_iv_poly_ascending": beta.tolist(),
            "min_log_forward_moneyness": float(x.min()), "max_log_forward_moneyness": float(x.max()),
            "quote_count": len(good), "unique_strikes": len(set(r["strike"] for r in good)),
            "weighted_rmse_iv": rmse, "median_abs_error_iv": float(np.median(np.abs(residual))),
            "max_abs_error_iv": float(np.max(np.abs(residual))), "fit_method": "weighted quadratic least squares in implied volatility"}


def _evaluate(smile: dict, k: float) -> float:
    z = k/smile["x_scale"]
    a, b, c = smile["coefficients_iv_poly_ascending"]
    return a+b*z+c*z*z


class CalibratedSurface:
    """Piecewise-tenor surface; linear interpolation is in total variance.

    No extrapolation is allowed outside calibrated tenor or common moneyness bounds.
    This quadratic smile is a diagnostic fit and is not guaranteed arbitrage-free.
    """
    def __init__(self, smiles: list[dict]):
        self.smiles = sorted(smiles, key=lambda s: s["time_years"])
        if len(self.smiles) < 2:
            raise ValueError("surface needs at least two calibrated expiries")

    def volatility(self, log_forward_moneyness: float, time_years: float) -> float:
        if not math.isfinite(log_forward_moneyness) or not math.isfinite(time_years) or time_years <= 0:
            raise ValueError("moneyness and positive tenor must be finite")
        ts = np.asarray([s["time_years"] for s in self.smiles])
        if time_years < ts[0] or time_years > ts[-1]:
            raise ValueError("tenor extrapolation is disabled")
        j = int(np.searchsorted(ts, time_years, side="left"))
        if j < len(ts) and abs(ts[j]-time_years) < 1e-12:
            brackets = [(self.smiles[j], 1.0)]
        else:
            left, right = self.smiles[j-1], self.smiles[j]
            w = (time_years-left["time_years"])/(right["time_years"]-left["time_years"])
            brackets = [(left, 1-w), (right, w)]
        total_var = 0.0
        for smile, weight in brackets:
            k = log_forward_moneyness
            if not smile["min_log_forward_moneyness"] <= k <= smile["max_log_forward_moneyness"]:
                raise ValueError("moneyness extrapolation is disabled")
            iv = _evaluate(smile, k)
            if not math.isfinite(iv) or iv <= 0:
                raise ValueError("calibrated fit returned nonpositive volatility")
            total_var += weight*smile["time_years"]*iv*iv
        return math.sqrt(total_var/time_years)


def load_surface(path: str | Path) -> CalibratedSurface:
    """Load a surface from the calibrated_smiles.csv file produced by this module."""
    smiles=[]
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["expiry_ms"]=int(row["expiry_ms"])
            for field in ("time_years","x_scale","min_log_forward_moneyness","max_log_forward_moneyness"):
                row[field]=float(row[field])
            row["coefficients_iv_poly_ascending"]=json.loads(row["coefficients_iv_poly_ascending"])
            smiles.append(row)
    return CalibratedSurface(smiles)


def fit_surface(rows: list[dict]) -> tuple[list[dict], dict]:
    expiries = sorted({r["expiry_ms"] for r in rows})
    smiles = [fit for expiry in expiries if (fit := _fit_smile([r for r in rows if r["expiry_ms"] == expiry]))]
    diagnostics = {"calibrated_expiries": len(smiles), "quote_count_fit": sum(s["quote_count"] for s in smiles),
                   "smile_weighted_rmse_iv_median": float(np.median([s["weighted_rmse_iv"] for s in smiles])) if smiles else None,
                   "smile_weighted_rmse_iv_max": float(max(s["weighted_rmse_iv"] for s in smiles)) if smiles else None,
                   "calendar_total_variance_violations": 0, "calendar_checks": 0,
                   "butterfly_arbitrage_free_guarantee": False}
    for first, second in zip(smiles, smiles[1:]):
        low=max(first["min_log_forward_moneyness"],second["min_log_forward_moneyness"])
        high=min(first["max_log_forward_moneyness"],second["max_log_forward_moneyness"])
        if high <= low: continue
        for k in np.linspace(low,high,41):
            w1=first["time_years"]*_evaluate(first,float(k))**2
            w2=second["time_years"]*_evaluate(second,float(k))**2
            diagnostics["calendar_checks"]+=1
            diagnostics["calendar_total_variance_violations"]+=int(w2+1e-10<w1)
    diagnostics["calendar_violation_fraction"]=(diagnostics["calendar_total_variance_violations"]/diagnostics["calendar_checks"]
        if diagnostics["calendar_checks"] else None)
    return smiles, diagnostics


def save_snapshot(out_dir: Path, context: dict, rows: list[dict]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    quote_path=out_dir/"chain_quotes.csv"
    fields=("symbol","side","strike","expiry_ms","time_years","unit","forward","log_forward_moneyness",
            "bid","ask","mid","relative_spread","last","volume_24h","trade_count_24h","trade_age_seconds",
            "mark_price","mark_iv","bid_iv_exchange","ask_iv_exchange","risk_free_rate","mid_iv",
            "iv_vs_exchange_mark","quote_status")
    with quote_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    smiles, diag=fit_surface(rows)
    surface_path=out_dir/"calibrated_smiles.csv"
    smile_fields=("expiry_date_utc","expiry_ms","time_years","quote_count","unique_strikes","x_scale",
                  "coefficients_iv_poly_ascending","weighted_rmse_iv","median_abs_error_iv","max_abs_error_iv",
                  "min_log_forward_moneyness","max_log_forward_moneyness","fit_method")
    with surface_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=smile_fields);writer.writeheader()
        for smile in smiles: writer.writerow(smile)
    grid_path=out_dir/"surface_grid.csv"
    with grid_path.open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(("expiry_date_utc","time_years","log_forward_moneyness","fitted_iv"))
        for smile in smiles:
            for k in np.linspace(smile["min_log_forward_moneyness"],smile["max_log_forward_moneyness"],41):
                writer.writerow((smile["expiry_date_utc"],smile["time_years"],k,_evaluate(smile,float(k))))
    eligible=[r for r in rows if r.get("quote_status")=="eligible" and math.isfinite(r.get("mid_iv",math.nan))]
    iv_diffs=np.asarray([r["iv_vs_exchange_mark"] for r in eligible if math.isfinite(r.get("iv_vs_exchange_mark",math.nan))])
    result={**context,"eligible_two_sided_iv_quotes":len(eligible),"chain_rows":len(rows),
            "expiry_count":len({r["expiry_ms"] for r in eligible}),
            "surface_fit":diag,
            "mid_quote_iv_vs_exchange_mark_iv":{
              "comparison_count":int(len(iv_diffs)),
              "median_abs_difference":float(np.median(np.abs(iv_diffs))) if len(iv_diffs) else None,
              "rmse":float(np.sqrt(np.mean(iv_diffs**2))) if len(iv_diffs) else None},
            "files":[quote_path.name,surface_path.name,grid_path.name],
            "scope_note":"Snapshot marks and top-of-book quotes are live at capture, not a historical chain. Quadratic smile fits are not guaranteed free of butterfly/calendar arbitrage."}
    (out_dir/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description="Capture and calibrate a Binance BTC options snapshot")
    parser.add_argument("--out",type=Path,default=Path("data/binance_options_snapshot"))
    args=parser.parse_args()
    context,rows=fetch_btc_chain()
    enrich_quotes(context,rows)
    result=save_snapshot(args.out,context,rows)
    print(json.dumps(result,indent=2,allow_nan=False))
    print(f"\nSaved market snapshot and calibration to {args.out.resolve()}")


if __name__=="__main__":
    main()
