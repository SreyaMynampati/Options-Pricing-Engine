"""Vectorized European Black--Scholes and implied volatility utilities.

Rates, dividend yields, and volatilities are annualized decimals; time is in years.
Inputs broadcast using NumPy rules. This module is a research/reference implementation;
market calendars, quote cleaning, and surface interpolation belong in caller code.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _inputs(spot: ArrayLike, strike: ArrayLike, time: ArrayLike,
            rate: ArrayLike, dividend_yield: ArrayLike, volatility: ArrayLike):
    s, k, t, r, q, v = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64) for x in
          (spot, strike, time, rate, dividend_yield, volatility)])
    if not all(np.all(np.isfinite(x)) for x in (s, k, t, r, q, v)):
        raise ValueError("all inputs must be finite")
    if np.any(s <= 0) or np.any(k <= 0) or np.any(t < 0) or np.any(v < 0):
        raise ValueError("spot/strike must be positive; time/volatility non-negative")
    return s, k, t, r, q, v


def _cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # Abramowitz-Stegun approximation (absolute error below 8e-8), array-native.
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419*z)
    poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
    positive = 1.0 - np.exp(-0.5*z*z)/np.sqrt(2*np.pi)*poly
    return np.where(x >= 0, positive, 1.0-positive)


def black_scholes(spot: ArrayLike, strike: ArrayLike, time: ArrayLike,
                  rate: ArrayLike, volatility: ArrayLike, *,
                  option: str = "call", dividend_yield: ArrayLike = 0.0,
                  greeks: bool = False):
    """Price European calls/puts; optionally return price and standard Greeks.

    Vega and rho are derivatives per 1.00 change in volatility and rate.
    Theta is calendar-time decay per year. At expiry or zero volatility, price
    uses the deterministic discounted payoff and Greeks are reported as zero.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    s, k, t, r, q, v = _inputs(spot, strike, time, rate, dividend_yield, volatility)
    df_r, df_q = np.exp(-r*t), np.exp(-q*t)
    sign = 1.0 if option == "call" else -1.0
    live = (t > 0) & (v > 0)
    root = np.sqrt(np.where(live, t, 1.0))
    vsqrt = np.where(live, v*root, 1.0)
    d1 = (np.log(s/k) + (r-q+0.5*v*v)*t) / vsqrt
    d2 = d1-vsqrt
    price = sign*(s*df_q*_cdf(sign*d1)-k*df_r*_cdf(sign*d2))
    forward = s*np.exp((r-q)*t)
    deterministic = df_r*np.maximum(sign*(forward-k), 0.0)
    price = np.where(live, price, deterministic)
    if not greeks:
        return price.item() if price.ndim == 0 else price
    density = np.exp(-0.5*d1*d1)/np.sqrt(2*np.pi)
    delta = sign*df_q*_cdf(sign*d1)
    gamma = df_q*density/(s*vsqrt)
    vega = s*df_q*density*root
    theta = -s*df_q*density*v/(2*root) - sign*r*k*df_r*_cdf(sign*d2) + sign*q*s*df_q*_cdf(sign*d1)
    rho = sign*k*t*df_r*_cdf(sign*d2)
    values = [price, delta, gamma, vega, theta, rho]
    values = [np.where(live, x, 0.0) for x in values]
    return tuple(x.item() if x.ndim == 0 else x for x in values)


def implied_volatility(market_price: ArrayLike, spot: ArrayLike, strike: ArrayLike,
                       time: ArrayLike, rate: ArrayLike, *, option: str = "call",
                       dividend_yield: ArrayLike = 0.0, price_tolerance: float = 1e-10,
                       max_iterations: int = 100):
    """Safeguarded vectorized Newton/bisection solver; invalid quotes return NaN.

    No-arbitrage bounds are checked per contract. This is a European solver.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if price_tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid solver tolerance or iteration count")
    s, k, t, r, q, dummy = _inputs(spot, strike, time, rate, dividend_yield, 0.0)
    market = np.broadcast_to(np.asarray(market_price, dtype=float), s.shape)
    if not np.all(np.isfinite(market)):
        raise ValueError("market prices must be finite")
    sign = 1.0 if option == "call" else -1.0
    lower = np.maximum(sign*(s*np.exp(-q*t)-k*np.exp(-r*t)), 0.0)
    upper = s*np.exp(-q*t) if option == "call" else k*np.exp(-r*t)
    valid = (t > 0) & (market > lower) & (market < upper) & (market >= 0)
    lo, hi = np.full(s.shape, 1e-8), np.full(s.shape, 0.5)
    for _ in range(8):
        too_low = black_scholes(s, k, t, r, hi, option=option, dividend_yield=q) < market
        hi = np.where(too_low, hi*2, hi)
    sigma = np.clip(np.full(s.shape, 0.2), lo, hi)
    for _ in range(max_iterations):
        price, _, _, vega, _, _ = black_scholes(s, k, t, r, sigma, option=option,
                                                 dividend_yield=q, greeks=True)
        error = price-market
        done = np.abs(error) <= price_tolerance
        hi = np.where((error > 0) & ~done, sigma, hi)
        lo = np.where((error <= 0) & ~done, sigma, lo)
        candidate = sigma-error/np.maximum(vega, 1e-14)
        safe = np.isfinite(candidate) & (candidate > lo) & (candidate < hi) & (vega >= 1e-14)
        sigma = np.where(done, sigma, np.where(safe, candidate, 0.5*(lo+hi)))
        if np.all(done | ~valid):
            break
    residual = np.abs(black_scholes(s, k, t, r, sigma, option=option, dividend_yield=q)-market)
    out = np.where(valid & (residual <= max(price_tolerance, 1e-8)), sigma, np.nan)
    return out.item() if out.ndim == 0 else out
