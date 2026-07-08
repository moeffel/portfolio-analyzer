"""Portfolio-level structure: weights, concentration, diversification, allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weights_from_values(values: pd.Series) -> pd.Series:
    """Normalise market values into weights summing to 1."""
    total = values.sum()
    if total <= 0:
        raise ValueError("Total portfolio value must be positive")
    return values / total


def herfindahl(weights: pd.Series) -> float:
    """Herfindahl-Hirschman concentration index (sum of squared weights), 1/N..1."""
    w = weights.dropna()
    return float(np.square(w).sum())


def effective_holdings(weights: pd.Series) -> float:
    """Effective number of positions = 1 / HHI. N for equal weights, lower if concentrated."""
    hhi = herfindahl(weights)
    return float(1.0 / hhi) if hhi > 0 else float("nan")


def allocation_by_class(holdings: pd.DataFrame, weight_col: str = "weight",
                        class_col: str = "asset_class") -> pd.Series:
    """Aggregate weights by asset class label."""
    if class_col not in holdings:
        return pd.Series(dtype=float)
    return holdings.groupby(class_col)[weight_col].sum().sort_values(ascending=False)


def allocation_drift(current: pd.Series, target: dict) -> pd.DataFrame:
    """Per-class current vs target weight and drift.

    Returns a frame indexed by class with columns [current, target, drift].
    Classes present in either side are included.
    """
    classes = sorted(set(current.index) | set(target.keys()))
    rows = []
    for c in classes:
        cur = float(current.get(c, 0.0))
        tgt = float(target.get(c, 0.0))
        rows.append({"asset_class": c, "current": cur, "target": tgt, "drift": cur - tgt})
    return pd.DataFrame(rows).set_index("asset_class")


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix across assets (pairwise complete)."""
    return asset_returns.corr()


def average_pairwise_correlation(asset_returns: pd.DataFrame) -> float:
    """Mean off-diagonal correlation — a crude diversification indicator."""
    corr = asset_returns.corr().values
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    off = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off))


def diversification_summary(weights: pd.Series, asset_returns: pd.DataFrame | None = None) -> dict:
    out = {
        "n_positions": int(weights.dropna().shape[0]),
        "hhi": herfindahl(weights),
        "effective_holdings": effective_holdings(weights),
        "largest_position": float(weights.max()) if len(weights) else float("nan"),
    }
    if asset_returns is not None and asset_returns.shape[1] >= 2:
        out["avg_pairwise_corr"] = average_pairwise_correlation(asset_returns)
    return out
