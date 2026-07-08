"""Common data container returned by every adapter.

Keeping one plain container means the CLI, dashboard and analytics never care
whether the numbers came from yfinance, a CSV, or the synthetic sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MarketData:
    """Everything the analytics need for one run.

    prices : wide DataFrame, DatetimeIndex, columns = tickers (adjusted close).
    fundamentals : DataFrame indexed by ticker, fundamental columns.
    holdings : DataFrame with at least [ticker, value]; optional
        [asset_class, ter, region, quantity, price].
    benchmark_prices : optional Series of the benchmark's adjusted close.
    meta : free-form provenance dict (source, fetched_at, warnings).
    """

    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_prices: pd.Series | None = None
    meta: dict = field(default_factory=dict)

    def tickers(self) -> list[str]:
        if not self.holdings.empty and "ticker" in self.holdings:
            return list(self.holdings["ticker"])
        return list(self.prices.columns)

    def weights(self) -> pd.Series:
        """Normalised weights from holdings 'value' (or 'weight') column, indexed by ticker."""
        h = self.holdings
        if h.empty:
            return pd.Series(dtype=float)
        h = h.set_index("ticker")
        if "weight" in h:
            w = h["weight"].astype(float)
        elif "value" in h:
            w = h["value"].astype(float)
        else:
            raise ValueError("holdings needs a 'value' or 'weight' column")
        total = w.sum()
        return w / total if total > 0 else w
