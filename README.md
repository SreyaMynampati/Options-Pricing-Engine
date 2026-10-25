<div align="center">

# VOLTERRA
### Stochastic Models & Numerical Methods for Derivative Pricing

**Risk-neutral valuation · Greeks · optimal stopping · implied-volatility calibration**

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](#build-and-run)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#python-workflows)
[![NumPy](https://img.shields.io/badge/NumPy-vectorized-013243?logo=numpy&logoColor=white)](#python-workflows)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

A mathematical-finance research engine for derivative valuation under stochastic models. Its implemented methods span Black–Scholes risk-neutral valuation, sensitivity analysis, lattice and regression-based optimal stopping, Monte Carlo estimation, safeguarded nonlinear root-finding, and empirical volatility-surface calibration. The C++23 core and NumPy tools are paired with Binance BTC market data and reproducible numerical results.

For beginners: an option is a contract whose value depends on an underlying asset such as Bitcoin. Give the engine the asset price, strike, expiry, interest rate, and volatility, and its pricing models estimate the option's value today. Greeks describe how that estimate responds to changes in market inputs; implied volatility reverses the calculation to find the volatility consistent with an observed quote.

For a European payoff $H(S_T)$, the risk-neutral valuation implemented by the engine is $V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[H(S_T)]$. The project compares analytic valuation with tree and Monte Carlo estimates, and solves $V_{BS}(\sigma)=V_{market}$ for implied volatility.

</div>

## What it does

| Component | Implemented behavior |
|---|---|
| C++23 pricing | European Black–Scholes–Merton price and Greeks; Cox–Ross–Rubinstein tree; antithetic European Monte Carlo; Longstaff–Schwartz American Monte Carlo |
| Implied volatility | Safeguarded Newton iteration with a maintained bracket and bisection fallback; theoretical price-bound checks |
| Python analytics | NumPy-broadcasting Black–Scholes prices, Greeks, and implied volatility |
| Historical underlying data | Downloads Binance BTCUSDT daily spot candles, calculates trailing realized volatility, and creates lagged illustrative next-open marks |
| Live options snapshot | Captures Binance BTC option contracts, index price, top-of-book bid/ask, mark prices, and exchange mark IV |
| Volatility surface | Solves IV from two-sided quote midpoints, fits a weighted quadratic smile per expiry, and interpolates total variance between expiries without extrapolating |

## Measured results

The saved market-data result files were generated from Binance's public APIs. The benchmark values below come from a local Release build; source timestamps and assumptions are recorded with the data.

| Result | Measurement |
|---|---:|
| Historical BTCUSDT spot candles | 3,356 daily candles, 2017-08-17 to 2026-10-24; the requested ten-year window starts earlier than the pair's available history |
| Trailing 30-day realized volatility on that spot sample | Median 55.04%; range 16.76%–211.78% |
| Binance BTC options snapshot | 2026-10-25 17:00:06 UTC; index 80,811.75; 622 active BTCUSDT contracts joined across 11 expiries |
| Two-sided option quotes with finite midpoint IV | 605 contracts |
| Weighted smile fit error | Median 0.600 volatility points across expiries; worst expiry 2.101 points |
| Calendar total-variance check | 1 violation across 410 adjacent-expiry/moneyness checks (0.24%); the fitted surface is **not guaranteed arbitrage-free** |
| Saved surface query | 38.57% IV at ATM log-forward-moneyness and 0.25-year tenor |
| Midpoint-derived IV vs Binance mark IV | Median absolute difference 0.784 volatility points; RMSE 7.778 points, reflecting noisy and wide quotes in the snapshot |
| C++ Black–Scholes throughput | 12.59 million varied contract prices/second; 79.412 ns/price median loop time |
| C++ 512-step American CRR put | 6.985 ms median per price; price 6.691 |
| C++ antithetic Monte Carlo | 250,000 paths in 9.018 ms median; estimate 8.081, standard error 0.019 |
| C++ American Longstaff–Schwartz Monte Carlo | 50,000 paths × 50 steps in 324.081 ms median; estimate 6.665, standard error 0.020 |

These are the results from the run shown above, using a Release build. Black–Scholes throughput is the median of seven timed one-million-contract loops after a warm-up. Tree and Monte Carlo timings are medians over repeated runs. Benchmark results vary by machine and build settings; these numbers are measurements, not a portable latency guarantee.

The C++ demo's European call price is **5.548166**. Its 500,000-path Monte Carlo estimate is **5.536364**, with reported 95% interval **[5.512437, 5.560290]**. The same demo recovers its input volatility of **25.0000%** from the analytic price. Validation checks the canonical one-year ATM call (**10.4505835722**), put-call parity, finite-difference Greeks, tree convergence, American exercise bounds, solver round-trips, invalid inputs, and deterministic Monte Carlo seeding. Longstaff–Schwartz tests compare an American put to a 1,000-step CRR reference and an American call without dividends to its Black–Scholes value.

## Build and run

Requires CMake 3.20+ and a C++23 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/volterra_demo
```

Run the C++ and Python validation suites:

```bash
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run the local C++ benchmark:

```bash
./build/volterra_bench
```

## Python workflows

Install the project and its NumPy dependency:

```bash
python -m pip install -e .
```

Vectorized price and IV example:

```python
import numpy as np
from volterra import black_scholes, implied_volatility

spots = np.array([95.0, 100.0, 105.0])
prices = black_scholes(spots, 100.0, 30/365, 0.045, 0.30, option="call")
vols = implied_volatility(prices, spots, 100.0, 30/365, 0.045, option="call")
print(prices)
print(vols)  # approximately [0.30, 0.30, 0.30]
```

With `greeks=True`, `black_scholes` returns `(price, delta, gamma, vega, theta, rho)`. Vega and rho are per 1.00 change in volatility and rate; theta is calendar-time decay per year.

Load the saved fitted surface and query a volatility at log-forward-moneyness `k` and tenor `T` (years):

```python
from volterra.options_market import load_surface

surface = load_surface("data/binance_options_snapshot/calibrated_smiles.csv")
iv = surface.volatility(log_forward_moneyness=0.0, time_years=0.25)
print(iv)
```

Queries outside the fitted maturity or moneyness ranges raise `ValueError`; the surface does not extrapolate.

## Reproduce the Binance data

Download the latest completed daily BTCUSDT spot history and produce the historical study:

```bash
python -m volterra.binance --symbol BTCUSDT --years 10 --out data/binance_btcusdt_10y
```

Capture and calibrate a current BTC options market snapshot:

```bash
python -m volterra.options_market --out data/binance_options_snapshot
```

Both commands use public, unauthenticated Binance market data. The spot downloader uses daily klines. The options workflow combines exchange contract information, the BTCUSDT index, option mark data, and option ticker bid/ask quotes. Binance documents the [Options market-data endpoints](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-options/api/rest-api/market-data) and [public Spot market-data endpoints](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md).

## Result files

- `data/binance_btcusdt_10y/binance_spot_daily.csv` — raw daily BTCUSDT spot OHLCV.
- `data/binance_btcusdt_10y/illustrative_options.csv` — prior-close realized-volatility signals and hypothetical next-open Black–Scholes marks.
- `data/binance_btcusdt_10y/metadata.json` and `bias_audit.json` — date coverage, assumptions, and bias notes.
- `data/binance_options_snapshot/chain_quotes.csv` — captured BTC option chain with bid, ask, midpoint, mark, mark IV, and midpoint-derived IV where eligible.
- `data/binance_options_snapshot/calibrated_smiles.csv` — per-expiry fitted smile coefficients and errors.
- `data/binance_options_snapshot/surface_grid.csv` — fitted IV values across each expiry's observed moneyness range.
- `data/binance_options_snapshot/results.json` — capture time, quote counts, surface-fit diagnostics, and comparisons to Binance marks.

## Models and conventions

- Spot and strike are positive prices. Time is a year fraction using 365 days for crypto option expiries. Rates, yields, and volatility are annualized decimals (`0.20` means 20%).
- Black–Scholes supports European exercise. At expiry or zero volatility, it returns the deterministic discounted payoff; Greeks are zero at these degenerate points.
- The CRR tree supports European and American calls and puts with early exercise at each node. It rejects invalid risk-neutral probabilities instead of silently returning a price.
- European Monte Carlo uses terminal payoffs. American Monte Carlo uses Longstaff–Schwartz regression with a quadratic basis in log-moneyness. Both use antithetic paths and a normal-approximation 95% interval based on antithetic pair averages; for the American estimator, this interval does not include uncertainty from estimating the exercise policy.
- IV inversion uses market price midpoint, BTC option index, exchange-reported risk-free rate, zero continuous yield, and time from the index timestamp to contract expiry.
- The surface fits a weighted quadratic in implied volatility against log-forward-moneyness for each expiry. Tenor interpolation is linear in total variance. The fit is diagnostic, has no butterfly-arbitrage guarantee, and reports calendar total-variance violations instead of hiding them.

## Data coverage

The BTCUSDT study contains daily spot candles from 2017-08-17 through 2026-10-24. Its next-open option marks are model estimates using lagged realized volatility. The options calibration uses the timestamped Binance snapshot in the results table: 622 active contracts and 605 two-sided quotes across 11 expiries. The surface diagnostics and endpoint capture times are saved with the snapshot. The C++ pricing engine and Python analytics are separate interfaces.

## License

MIT. See [LICENSE](LICENSE).
