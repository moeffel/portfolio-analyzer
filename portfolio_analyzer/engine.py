"""Analysis engine — ties data + config into a structured AnalysisResult.

This is the single entry point the CLI and dashboard call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import AnalysisConfig
from .data.base import MarketData
from .analytics import returns as R
from .analytics import risk as Risk
from .analytics import factors as F
from .analytics import portfolio as P
from .analytics import recommend as Rec


@dataclass
class AnalysisResult:
    metrics: dict = field(default_factory=dict)
    benchmark_metrics: dict | None = None
    diversification: dict = field(default_factory=dict)
    allocation: pd.Series = field(default_factory=pd.Series)
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.Series = field(default_factory=pd.Series)
    factor_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    flags: list = field(default_factory=list)
    portfolio_return_series: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    meta: dict = field(default_factory=dict)


def analyze(data: MarketData, cfg: AnalysisConfig | None = None) -> AnalysisResult:
    cfg = cfg or AnalysisConfig()
    ppy = cfg.trading_days

    weights = data.weights()
    holdings = data.holdings.copy()
    if "weight" not in holdings and not weights.empty:
        holdings = holdings.merge(
            weights.rename("weight"), left_on="ticker", right_index=True, how="left"
        )

    # --- returns / risk ---
    metrics, benchmark_metrics = {}, None
    port_rets = pd.Series(dtype=float)
    equity_curve = pd.Series(dtype=float)
    corr = pd.DataFrame()

    if not data.prices.empty:
        asset_rets = R.to_returns(data.prices)
        aligned_w = weights.reindex(asset_rets.columns).dropna()
        if not aligned_w.empty:
            port_rets = R.portfolio_returns(asset_rets, aligned_w)
            equity_curve = (1.0 + port_rets).cumprod()
        corr = P.correlation_matrix(asset_rets)

        bench_rets = None
        if data.benchmark_prices is not None:
            bench_rets = R.to_returns(data.benchmark_prices)

        if not port_rets.empty:
            metrics = Risk.summary(
                port_rets, mar=cfg.mar, risk_free=cfg.risk_free,
                periods_per_year=ppy, benchmark_returns=bench_rets,
            )
        if bench_rets is not None and not bench_rets.empty:
            benchmark_metrics = Risk.summary(
                bench_rets, mar=cfg.mar, risk_free=cfg.risk_free, periods_per_year=ppy,
            )

    # --- structure / allocation ---
    alloc = P.allocation_by_class(holdings) if "asset_class" in holdings else pd.Series(dtype=float)
    drift = P.allocation_drift(alloc, cfg.target_allocation) if not alloc.empty else pd.DataFrame()
    div = {}
    if not weights.empty:
        asset_rets_for_div = R.to_returns(data.prices) if not data.prices.empty else None
        div = P.diversification_summary(weights, asset_rets_for_div)

    # --- factor scores ---
    scores = pd.DataFrame()
    if not data.fundamentals.empty:
        scores = F.factor_scores(
            data.fundamentals,
            prices=data.prices if not data.prices.empty else None,
            weights=cfg.normalized_factor_weights(),
            periods_per_year=ppy,
        )

    # --- diagnostics ---
    flags = Rec.diagnose(
        weights=weights,
        alloc_by_class=alloc,
        drift=drift,
        diversification=div,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        holdings=holdings,
        cfg=cfg,
    )

    return AnalysisResult(
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        diversification=div,
        allocation=alloc,
        drift=drift,
        weights=weights,
        factor_scores=scores,
        correlation=corr,
        flags=flags,
        portfolio_return_series=port_rets,
        equity_curve=equity_curve,
        meta={**data.meta, "config_mar": cfg.mar, "config_rf": cfg.risk_free},
    )
