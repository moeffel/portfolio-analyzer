"""Offline CSV/JSON adapter — fully reproducible, no network.

Expected files (all UTF-8):

holdings.csv   ticker,value,asset_class,ter,region      (value or weight required)
prices.csv     date,<TICKER1>,<TICKER2>,...             (wide, adjusted close)
fundamentals.csv  ticker,pe,pb,fcf_yield,roe,gross_margin,debt_to_equity,market_cap
                                                          (optional; any subset of columns)

Prices in "long" form (date,ticker,close) are auto-pivoted.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import MarketData


def _read_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    # long form?
    if {"ticker", "close"}.issubset({c.lower() for c in df.columns}):
        tcol = cols.get("ticker")
        ccol = cols.get("close")
        wide = df.pivot(index=date_col, columns=tcol, values=ccol)
    else:
        wide = df.set_index(date_col)
    wide = wide.sort_index()
    wide.index.name = "date"
    return wide.astype(float)


def load_csv(
    holdings_path: str | Path,
    prices_path: str | Path | None = None,
    fundamentals_path: str | Path | None = None,
    benchmark: str | None = None,
) -> MarketData:
    holdings = pd.read_csv(holdings_path)
    holdings.columns = [c.strip().lower() for c in holdings.columns]
    if "ticker" not in holdings:
        raise ValueError("holdings CSV must have a 'ticker' column")

    prices = pd.DataFrame()
    benchmark_prices = None
    if prices_path:
        prices = _read_prices(Path(prices_path))
        if benchmark and benchmark in prices.columns:
            benchmark_prices = prices[benchmark]
            prices = prices.drop(columns=[benchmark])

    fundamentals = pd.DataFrame()
    if fundamentals_path:
        fundamentals = pd.read_csv(fundamentals_path)
        fundamentals.columns = [c.strip().lower() for c in fundamentals.columns]
        fundamentals = fundamentals.set_index("ticker")

    return MarketData(
        prices=prices,
        fundamentals=fundamentals,
        holdings=holdings,
        benchmark_prices=benchmark_prices,
        meta={"source": "csv", "holdings_path": str(holdings_path)},
    )
