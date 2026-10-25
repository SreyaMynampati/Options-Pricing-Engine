"""Vectorized NumPy reference tools for the Volterra options engine."""
from .analytics import black_scholes, implied_volatility

__all__ = ["black_scholes", "implied_volatility"]
