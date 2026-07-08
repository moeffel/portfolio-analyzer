"""Return computations from price series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_returns(prices: pd.Series | pd.DataFrame, log: bool = False):
    """Simple (or log) periodic returns from a price series/frame.

    NaNs from the initial diff are dropped. Works on a Series (one asset) or a
    DataFrame (columns = assets).
    """
    if log:
        r = np.log(prices / prices.shift(1))
    else:
        r = prices.pct_change()
    return r.dropna(how="all")


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the series."""
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Geometric (CAGR-style) annualised return from periodic returns."""
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:  # wiped out — avoid complex roots
        return -1.0
    return growth ** (periods_per_year / n) - 1.0


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict | pd.Series) -> pd.Series:
    """Weighted portfolio return series (weights held constant / rebalanced each period).

    Weights are aligned to the frame's columns and renormalised over the
    assets that are actually present.
    """
    w = pd.Series(weights, dtype=float)
    cols = [c for c in asset_returns.columns if c in w.index]
    if not cols:
        raise ValueError("No overlap between weights and return columns")
    w = w[cols]
    w = w / w.sum()
    return asset_returns[cols].fillna(0.0).mul(w, axis=1).sum(axis=1)
