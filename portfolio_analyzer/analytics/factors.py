"""Cross-sectional factor scoring.

Scores are computed as z-scores *across the supplied universe* — a single
stock in isolation has no meaningful factor score, so a peer set is required.
Higher composite = stronger evidence-based tilt (value + quality + momentum +
size + low-vol). This is deliberately a *tilt-ranking*, not a buy signal:

    RESEARCH CAVEAT (McLean & Pontiff 2016): published anomaly returns decay
    ~26% out-of-sample and ~58% post-publication, and residual premia sit in
    illiquid, hard-to-arbitrage stocks. Treat high scores as modest,
    cost-aware tilts — never as guaranteed premia.

Expected fundamental columns (all optional; missing ones are skipped for that
factor): pe, pb, fcf_yield, roe, gross_margin, debt_to_equity, market_cap.
Momentum and low-vol are derived from a price history if provided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_CAVEAT = (
    "Faktor-Scores sind cross-sektionale Tilt-Rankings, keine Kaufsignale. "
    "Publizierte Prämien verlieren ~26% out-of-sample / ~58% post-publication "
    "(McLean & Pontiff 2016) und stecken in illiquiden Titeln. Moderat gewichten."
)


def _zscore(s: pd.Series) -> pd.Series:
    """Robust-ish z-score; constant/empty series -> zeros."""
    s = s.astype(float)
    valid = s.dropna()
    if len(valid) < 2 or valid.std(ddof=0) == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - valid.mean()) / valid.std(ddof=0)
    return z.clip(-3, 3)  # winsorise extreme outliers


def _winsor_mean(cols: list[pd.Series]) -> pd.Series:
    """Average of available z-score components row-wise, ignoring NaNs."""
    if not cols:
        return pd.Series(dtype=float)
    df = pd.concat(cols, axis=1)
    return df.mean(axis=1, skipna=True)


def momentum_12_1(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """12-1 month momentum: return over the last ~12m skipping the most recent ~1m."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    lookback = periods_per_year
    skip = periods_per_year // 12
    out = {}
    for col in prices.columns:
        p = prices[col].dropna()
        if len(p) < lookback + 1:
            out[col] = np.nan
            continue
        end = p.iloc[-(skip + 1)]
        start = p.iloc[-(lookback + 1)]
        out[col] = end / start - 1.0 if start > 0 else np.nan
    return pd.Series(out)


def trailing_vol(prices: pd.DataFrame, window: int = 252, periods_per_year: int = 252) -> pd.Series:
    """Annualised trailing volatility per asset (for the low-vol factor)."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    rets = prices.pct_change()
    out = {}
    for col in prices.columns:
        r = rets[col].dropna().iloc[-window:]
        out[col] = r.std(ddof=1) * np.sqrt(periods_per_year) if len(r) > 1 else np.nan
    return pd.Series(out)


def factor_scores(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    weights: dict | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute per-factor z-scores and a weighted composite over a universe.

    Parameters
    ----------
    fundamentals : DataFrame indexed by ticker with fundamental columns.
    prices : optional wide price frame (columns = tickers) for momentum/low-vol.
    weights : optional factor weights (falls back to equal over available factors).

    Returns a DataFrame indexed by ticker with columns
    [value, quality, momentum, size, low_vol, composite].
    """
    f = fundamentals.copy()
    idx = f.index

    # --- Value: cheaper = higher score ---
    value_parts = []
    if "pe" in f:
        value_parts.append(_zscore(1.0 / f["pe"].where(f["pe"] > 0)))  # earnings yield
    if "pb" in f:
        value_parts.append(_zscore(1.0 / f["pb"].where(f["pb"] > 0)))  # book-to-price
    if "fcf_yield" in f:
        value_parts.append(_zscore(f["fcf_yield"]))
    value = _winsor_mean(value_parts).reindex(idx)

    # --- Quality: profitable, low leverage = higher score ---
    quality_parts = []
    if "roe" in f:
        quality_parts.append(_zscore(f["roe"]))
    if "gross_margin" in f:
        quality_parts.append(_zscore(f["gross_margin"]))
    if "debt_to_equity" in f:
        quality_parts.append(_zscore(-f["debt_to_equity"]))  # less debt = better
    quality = _winsor_mean(quality_parts).reindex(idx)

    # --- Momentum: 12-1 ---
    if prices is not None and not prices.empty:
        mom_raw = momentum_12_1(prices, periods_per_year).reindex(idx)
    elif "momentum_12_1" in f:
        mom_raw = f["momentum_12_1"]
    else:
        mom_raw = pd.Series(np.nan, index=idx)
    momentum = _zscore(mom_raw)

    # --- Size: smaller = higher score (size premium) ---
    if "market_cap" in f:
        size = _zscore(-np.log(f["market_cap"].where(f["market_cap"] > 0)))
    else:
        size = pd.Series(0.0, index=idx)

    # --- Low volatility: lower vol = higher score ---
    if prices is not None and not prices.empty:
        vol_raw = trailing_vol(prices, periods_per_year=periods_per_year).reindex(idx)
    elif "trailing_vol" in f:
        vol_raw = f["trailing_vol"]
    else:
        vol_raw = pd.Series(np.nan, index=idx)
    low_vol = _zscore(-vol_raw)

    scores = pd.DataFrame(
        {
            "value": value,
            "quality": quality,
            "momentum": momentum,
            "size": size,
            "low_vol": low_vol,
        }
    ).fillna(0.0)

    w = weights or {c: 1.0 for c in scores.columns}
    w = {k: v for k, v in w.items() if k in scores.columns}
    wsum = sum(w.values()) or 1.0
    composite = sum(scores[k] * (v / wsum) for k, v in w.items())
    scores["composite"] = composite
    return scores.sort_values("composite", ascending=False)
