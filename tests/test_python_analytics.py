import unittest
import sys
from pathlib import Path

import numpy as np

# Keep the test command runnable from a fresh checkout without requiring an
# editable package install first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from volterra import black_scholes, implied_volatility


class BlackScholesTests(unittest.TestCase):
    def test_reference_prices_and_put_call_parity(self):
        call = black_scholes(100, 100, 1, .05, .2)
        put = black_scholes(100, 100, 1, .05, .2, option="put")
        self.assertAlmostEqual(call, 10.4505835722, delta=2e-5)
        self.assertAlmostEqual(put, 5.5735260223, delta=2e-5)
        self.assertAlmostEqual(call-put, 100-100*np.exp(-.05), places=6)

    def test_broadcasting_and_greeks(self):
        spots = np.array([95., 100., 105.])
        prices = black_scholes(spots, 100, 30/365, .04, .25)
        self.assertEqual(prices.shape, (3,))
        values = black_scholes(100, 100, 1, .05, .2, greeks=True)
        self.assertEqual(len(values), 6)
        self.assertAlmostEqual(values[1], .6368305905, delta=2e-6)
        self.assertGreater(values[2], 0)
        self.assertGreater(values[3], 0)

    def test_zero_volatility_and_expiry(self):
        price = black_scholes(100, 95, 1, .05, 0, dividend_yield=.02)
        expected = np.exp(-.05)*max(100*np.exp(.03)-95, 0)
        self.assertAlmostEqual(price, expected, places=10)
        self.assertEqual(black_scholes(90, 100, 0, .05, .2, option="put"), 10)

    def test_implied_volatility_round_trip_and_invalid_quotes(self):
        spots = np.array([90., 100., 110.])
        for option in ("call", "put"):
            for vol in (.08, .2, .65, 1.5):
                prices = black_scholes(spots, 100, .4, .025, vol, option=option, dividend_yield=.008)
                recovered = implied_volatility(prices, spots, 100, .4, .025,
                                               option=option, dividend_yield=.008)
                np.testing.assert_allclose(recovered, vol, atol=2e-7, rtol=0)
        self.assertTrue(np.isnan(implied_volatility(0, 100, 100, 1, .05)))
        self.assertTrue(np.isnan(implied_volatility(100, 100, 100, 1, .05)))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            black_scholes(-1, 100, 1, .05, .2)
        with self.assertRaises(ValueError):
            black_scholes(100, 100, 1, .05, .2, option="straddle")


if __name__ == "__main__":
    unittest.main()
