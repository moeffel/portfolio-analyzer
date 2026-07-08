"""Risk & risk-adjusted-return metrics.

The centrepiece is the PMPT lens: downside deviation relative to a personal
Minimum Acceptable Return (MAR), and the Sortino ratio — the metric the
research note argues is more honest than Sharpe for retail investors.

All ratios take *annualised* inputs where relevant and annualise volatility
via sqrt-time. Functions operate on periodic (usually daily) return Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import annualized_return


def volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Annualised standard deviation (total dispersion — the MPT risk measure)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, mar: float, periods_per_year: int) -> float:
    """Annualised downside deviation below a Minimum Acceptable Return.

    MAR is supplied *annualised* and converted to the periodic frequency. Only
    shortfalls below the MAR contribute; upside is treated as zero risk (the
    core PMPT idea). Denominator is the full N (not just the shortfall count),
    following the standard Sortino convention.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    periodic_mar = (1.0 + mar) ** (1.0 / periods_per_year) - 1.0
    shortfall = np.minimum(returns - periodic_mar, 0.0)
    dd_periodic = np.sqrt(np.mean(np.square(shortfall)))
    return float(dd_periodic * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: float, periods_per_year: int) -> float:
    """(Annualised excess return) / (annualised volatility)."""
    vol = volatility(returns, periods_per_year)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    ann_ret = annualized_return(returns, periods_per_year)
    return (ann_ret - risk_free) / vol


def sortino_ratio(returns: pd.Series, mar: float, periods_per_year: int) -> float:
    """(Annualised return - MAR) / (annualised downside deviation vs MAR).

    The PMPT-consistent counterpart to Sharpe.
    """
    dd = downside_deviation(returns, mar, periods_per_year)
    if not np.isfinite(dd) or dd == 0:
        return float("nan")
    ann_ret = annualized_return(returns, periods_per_year)
    return (ann_ret - mar) / dd


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """OLS beta of an asset/portfolio vs a benchmark (aligned on common dates)."""
    df = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    var_b = b.var(ddof=1)
    if var_b == 0:
        return float("nan")
    return float(a.cov(b) / var_b)


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float,
    periods_per_year: int,
) -> float:
    """(Annualised excess return) / beta — reward per unit of systematic risk."""
    b = beta(returns, benchmark_returns)
    if not np.isfinite(b) or b == 0:
        return float("nan")
    ann_ret = annualized_return(returns, periods_per_year)
    return (ann_ret - risk_free) / b


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the compounded equity curve (negative)."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, alpha: float = 0.95, method: str = "historical") -> float:
    """Periodic VaR at confidence `alpha` (returned as a negative number).

    method="historical": empirical quantile.
    method="gaussian":  parametric (mean + z*sigma).
    Interpret as: with prob `alpha`, the period loss is not worse than |VaR|.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    q = 1.0 - alpha
    if method == "gaussian":
        from scipy.stats import norm

        mu, sigma = returns.mean(), returns.std(ddof=1)
        return float(mu + norm.ppf(q) * sigma)
    return float(np.quantile(returns, q))


def conditional_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Expected Shortfall / CVaR: mean loss conditional on breaching the VaR.

    Coherent risk measure (subadditive) — tail-sensitive, preferred over VaR.
    Returned as a negative number.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    var = value_at_risk(returns, alpha, method="historical")
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def summary(
    returns: pd.Series,
    *,
    mar: float,
    risk_free: float,
    periods_per_year: int,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """One-shot metrics bundle for a return series."""
    out = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "volatility": volatility(returns, periods_per_year),
        "downside_deviation": downside_deviation(returns, mar, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, mar, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "n_obs": int(returns.dropna().shape[0]),
    }
    if benchmark_returns is not None:
        out["beta"] = beta(returns, benchmark_returns)
        out["treynor"] = treynor_ratio(returns, benchmark_returns, risk_free, periods_per_year)
    return out
