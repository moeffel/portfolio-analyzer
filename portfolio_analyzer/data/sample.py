"""Deterministic synthetic sample data for demos and tests.

NOT real market data — a fixed-seed multivariate GBM with plausible drifts,
vols and a correlation structure (equities correlated; bonds/gold weakly or
negatively correlated; crypto = high-vol, high-corr-to-equity risk asset, per
the research finding). Fundamentals are hand-set to give a spread of factor
scores. Use it to see the tool work end-to-end without network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MarketData

# ticker -> (name, asset_class, annual_drift, annual_vol, ter)
_UNIVERSE = {
    "WORLD": ("MSCI World ETF (core)", "equity", 0.08, 0.15, 0.0020),
    "EM":    ("EM ETF (satellite)",    "equity", 0.09, 0.20, 0.0030),
    "VAL":   ("Value factor ETF",      "equity", 0.085, 0.16, 0.0035),
    "SMALL": ("Small-cap ETF",         "equity", 0.095, 0.22, 0.0035),
    "QLTY":  ("Quality factor ETF",    "equity", 0.088, 0.14, 0.0035),
    "AGG":   ("Aggregate bond ETF",    "bond",   0.03, 0.05, 0.0010),
    "GOLD":  ("Gold ETC",              "gold",   0.04, 0.16, 0.0040),
    "BTC":   ("Bitcoin",               "crypto", 0.20, 0.65, 0.0000),
}


def _corr_matrix(order: list[str]) -> np.ndarray:
    """Plausible correlation structure among the sample assets."""
    idx = {t: i for i, t in enumerate(order)}
    n = len(order)
    c = np.eye(n)

    def setc(a, b, v):
        c[idx[a], idx[b]] = c[idx[b], idx[a]] = v

    equities = ["WORLD", "EM", "VAL", "SMALL", "QLTY"]
    for i, a in enumerate(equities):
        for b in equities[i + 1:]:
            setc(a, b, 0.80)
    for a in equities:
        setc(a, "AGG", -0.10)
        setc(a, "GOLD", 0.05)
        setc(a, "BTC", 0.60)  # research: BTC behaves like an integrated risk asset
    setc("AGG", "GOLD", 0.15)
    setc("AGG", "BTC", 0.00)
    setc("GOLD", "BTC", 0.20)
    return c


def make_sample(seed: int = 7, days: int = 252 * 5) -> MarketData:
    rng = np.random.default_rng(seed)
    order = list(_UNIVERSE)
    drifts = np.array([_UNIVERSE[t][2] for t in order])
    vols = np.array([_UNIVERSE[t][3] for t in order])

    corr = _corr_matrix(order)
    cov = np.outer(vols, vols) * corr
    L = np.linalg.cholesky(cov)

    dt = 1.0 / 252
    z = rng.standard_normal((days, len(order)))
    shocks = z @ L.T
    daily = (drifts - 0.5 * vols**2) * dt + shocks * np.sqrt(dt)
    log_prices = np.cumsum(daily, axis=0)
    prices = 100.0 * np.exp(log_prices)

    dates = pd.bdate_range("2020-01-01", periods=days)
    px = pd.DataFrame(prices, index=dates, columns=order)
    px.index.name = "date"

    benchmark_prices = px["WORLD"].copy()

    # hand-set fundamentals giving a spread of factor tilts (equities only)
    fundamentals = pd.DataFrame.from_dict(
        {
            "WORLD": dict(pe=20, pb=2.8, fcf_yield=0.04, roe=0.16, gross_margin=0.40, debt_to_equity=0.8, market_cap=5e11),
            "EM":    dict(pe=13, pb=1.6, fcf_yield=0.06, roe=0.13, gross_margin=0.32, debt_to_equity=1.1, market_cap=2e11),
            "VAL":   dict(pe=11, pb=1.2, fcf_yield=0.08, roe=0.14, gross_margin=0.35, debt_to_equity=0.9, market_cap=1.5e11),
            "SMALL": dict(pe=15, pb=1.5, fcf_yield=0.05, roe=0.11, gross_margin=0.30, debt_to_equity=1.3, market_cap=8e9),
            "QLTY":  dict(pe=24, pb=5.0, fcf_yield=0.035, roe=0.28, gross_margin=0.55, debt_to_equity=0.4, market_cap=6e11),
        },
        orient="index",
    )
    fundamentals.index.name = "ticker"

    holdings = pd.DataFrame(
        [
            ("WORLD", 45000),
            ("EM", 8000),
            ("VAL", 7000),
            ("SMALL", 5000),
            ("QLTY", 6000),
            ("AGG", 12000),
            ("GOLD", 4000),
            ("BTC", 8000),  # ~9% -> should trip the crypto alert
        ],
        columns=["ticker", "value"],
    )
    holdings["asset_class"] = holdings["ticker"].map(lambda t: _UNIVERSE[t][1])
    holdings["ter"] = holdings["ticker"].map(lambda t: _UNIVERSE[t][4])
    holdings["name"] = holdings["ticker"].map(lambda t: _UNIVERSE[t][0])
    # everything here is a fund/ETC except BTC (crypto) — used by the concentration check
    holdings["security_type"] = holdings["ticker"].map(
        lambda t: "crypto" if t == "BTC" else "etf"
    )

    return MarketData(
        prices=px,
        fundamentals=fundamentals,
        holdings=holdings,
        benchmark_prices=benchmark_prices,
        meta={"source": "sample", "seed": seed, "synthetic": True},
    )
