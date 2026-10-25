"""Download Binance public spot candles and produce an illustrative pricing study.

Example: python -m volterra.binance --symbol BTCUSDT --years 10 --out data/btcusdt
Only public daily spot klines are fetched; this is not historical options data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from .analytics import black_scholes

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
DAY_MS = 86_400_000
FIELDS = ("open_time", "open", "high", "low", "close", "volume",
          "close_time", "quote_volume", "trade_count", "taker_buy_base",
          "taker_buy_quote", "ignore")


def fetch_daily_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    while cursor <= end_ms:
        query = urlencode({"symbol": symbol.upper(), "interval": "1d", "startTime": cursor,
                           "endTime": end_ms, "limit": 1000})
        request = Request(f"{BASE_URL}?{query}", headers={"User-Agent": "volterra-research/0.1"})
        for attempt in range(6):
            try:
                with urlopen(request, timeout=20) as response:
                    batch = json.loads(response.read())
                break
            except HTTPError as exc:
                if exc.code not in (418, 429, 500, 502, 503, 504) or attempt == 5:
                    raise RuntimeError(f"Binance HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
                time.sleep(min(float(exc.headers.get("Retry-After", 1)), 30.0) * (attempt + 1))
            except (URLError, TimeoutError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Binance request failed: {exc}") from exc
                time.sleep(0.5 * (2**attempt))
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + DAY_MS
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance; stopping to avoid duplicate loop")
        cursor = next_cursor
        if len(batch) < 1000:
            break
    # Keep one row per UTC open time and only the requested window.
    return list({int(row[0]): row for row in rows if start_ms <= int(row[0]) <= end_ms}.values())


def write_study(rows: list[list], out_dir: Path, symbol: str, requested_start: datetime,
                requested_end: datetime, risk_free_rate: float) -> dict:
    if len(rows) < 32:
        raise RuntimeError(f"Only {len(rows)} daily candles were returned; not enough for the study")
    rows.sort(key=lambda row: int(row[0]))
    closes = np.asarray([float(row[4]) for row in rows])
    returns = np.full(closes.shape, np.nan)
    returns[1:] = np.log(closes[1:] / closes[:-1])
    # 30 daily log returns; annualized using crypto's 365-day trading year.
    realized_vol = np.full(closes.shape, np.nan)
    for i in range(30, len(closes)):
        realized_vol[i] = np.std(returns[i-29:i+1], ddof=1) * np.sqrt(365.0)
    # A signal is known only after candle t closes. Price a hypothetical option
    # at the next UTC day's open, using only volatility observable at t's close.
    next_open = np.full(closes.shape, np.nan)
    execution_time = np.full(closes.shape, -1, dtype=np.int64)
    for i in range(len(rows) - 1):
        if int(rows[i + 1][0]) - int(rows[i][0]) == DAY_MS:
            next_open[i] = float(rows[i + 1][1])
            execution_time[i] = int(rows[i + 1][0])
    valid = np.isfinite(realized_vol) & np.isfinite(next_open)
    atm_call = np.full(closes.shape, np.nan)
    atm_put = np.full(closes.shape, np.nan)
    if np.any(valid):
        atm_call[valid] = black_scholes(next_open[valid], next_open[valid], 30/365,
                                         risk_free_rate, realized_vol[valid], option="call")
        atm_put[valid] = black_scholes(next_open[valid], next_open[valid], 30/365,
                                        risk_free_rate, realized_vol[valid], option="put")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, study_path = out_dir / "binance_spot_daily.csv", out_dir / "illustrative_options.csv"
    ledger_path, audit_path = out_dir / "universe_ledger.csv", out_dir / "bias_audit.json"
    with raw_path.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FIELDS); writer.writerows(rows)
    with study_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("signal_date_utc", "signal_available_at_utc", "signal_close",
                         "trailing_30d_realized_vol_known_at_signal", "execution_date_utc",
                         "next_day_open", "hypothetical_atm_30d_call_at_next_open",
                         "hypothetical_atm_30d_put_at_next_open"))
        for i, row in enumerate(rows):
            signal_date = datetime.fromtimestamp(int(row[0])/1000, timezone.utc).date().isoformat()
            signal_available = datetime.fromtimestamp((int(row[6])+1)/1000, timezone.utc).isoformat()
            execution_date = (datetime.fromtimestamp(int(execution_time[i])/1000, timezone.utc).isoformat()
                              if execution_time[i] >= 0 else "")
            writer.writerow((signal_date, signal_available, closes[i],
                             realized_vol[i] if np.isfinite(realized_vol[i]) else "",
                             execution_date, next_open[i] if np.isfinite(next_open[i]) else "",
                             atm_call[i] if np.isfinite(atm_call[i]) else "",
                             atm_put[i] if np.isfinite(atm_put[i]) else ""))

    first_seen = datetime.fromtimestamp(int(rows[0][0])/1000, timezone.utc).isoformat()
    last_seen = datetime.fromtimestamp(int(rows[-1][0])/1000, timezone.utc).isoformat()
    with ledger_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("symbol", "first_observed_utc", "last_observed_utc", "delisting_status",
                         "selection_rule", "coverage_limit"))
        writer.writerow((symbol.upper(), first_seen, last_seen, "unknown; current pair observed",
                         "explicit single-symbol request; no cross-sectional ranking",
                         "not a historical point-in-time Binance universe"))

    audit = {
        "lookahead_bias": {
            "status": "lagged_for_illustrative_option_marks",
            "signal_information_cutoff": "The 30 daily log returns ending at signal_date_utc close.",
            "execution_assumption": "Next consecutive UTC daily candle open; signal feature is not recomputed with that candle.",
            "same_close_execution": "prohibited by this output schema",
            "limitations": "No option quote or strategy PnL exists in this study; marks are hypothetical model values.",
        },
        "survivorship_bias": {
            "status": "not_applicable_to_single_asset_descriptive_series; unresolved_for_multi_asset_inference",
            "universe_size": 1,
            "ledger_file": ledger_path.name,
            "limitations": "A present-day BTCUSDT history is not a delisted-inclusive, point-in-time Binance asset universe. Do not use it to claim cross-sectional strategy performance.",
            "required_for_universe_backtest": "Point-in-time symbol membership including delisted/renamed pairs, listing and delisting dates, and inclusion rules fixed as of each historical date.",
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    metadata = {
        "provider": "Binance public Spot REST API",
        "endpoint": BASE_URL,
        "symbol": symbol.upper(), "interval": "1d", "timezone": "UTC",
        "requested_start_utc": requested_start.isoformat(),
        "requested_end_utc": requested_end.isoformat(),
        "actual_first_candle_utc": datetime.fromtimestamp(int(rows[0][0])/1000, timezone.utc).isoformat(),
        "actual_last_candle_utc": datetime.fromtimestamp(int(rows[-1][0])/1000, timezone.utc).isoformat(),
        "candle_count": len(rows), "risk_free_rate_assumption": risk_free_rate,
        "option_study": "Illustrative 30-calendar-day ATM BSM marks at next consecutive UTC daily open, using trailing realized volatility available at prior close; not observed option quotes or implied volatility.",
        "files": [raw_path.name, study_path.name, ledger_path.name, audit_path.name],
        "bias_audit": audit,
    }
    if np.any(valid):
        metadata["realized_vol_30d_summary"] = {
            "last_signal_vol": float(realized_vol[np.flatnonzero(valid)[-1]]),
            "median": float(np.median(realized_vol[np.isfinite(realized_vol)])),
            "min": float(np.min(realized_vol[np.isfinite(realized_vol)])),
            "max": float(np.max(realized_vol[np.isfinite(realized_vol)])),
        }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance daily spot history and price an illustrative ATM option series")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance Spot pair (default BTCUSDT)")
    parser.add_argument("--years", type=int, default=10, help="requested lookback years")
    parser.add_argument("--out", type=Path, default=Path("data/binance_btcusdt_10y"))
    parser.add_argument("--risk-free-rate", type=float, default=0.0,
                        help="flat annual continuous rate for illustrative BSM prices (default 0; not historical curve data)")
    args = parser.parse_args()
    if args.years < 1:
        parser.error("--years must be positive")
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    # Exclude the still-forming current daily candle; only include closed UTC days.
    end = today_utc - timedelta(milliseconds=1)
    start = today_utc - timedelta(days=365 * args.years)
    rows = fetch_daily_klines(args.symbol, int(start.timestamp()*1000), int(end.timestamp()*1000))
    result = write_study(rows, args.out, args.symbol, start, end, args.risk_free_rate)
    print(json.dumps(result, indent=2))
    print(f"\nSaved data and study to {args.out.resolve()}")


if __name__ == "__main__":
    main()
