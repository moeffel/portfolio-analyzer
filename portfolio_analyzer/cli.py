"""Command-line interface.

Examples
--------
# Offline demo on synthetic sample data, writes a Markdown note into the vault:
python -m portfolio_analyzer.cli --sample --report wiki/journal/portfolio-analysis.md

# Your real portfolio from CSVs:
python -m portfolio_analyzer.cli --holdings my.csv --prices prices.csv \
    --fundamentals fund.csv --benchmark URTH --mar 0.02 --report out.md

# Live via yfinance (on your own machine):
python -m portfolio_analyzer.cli --holdings my.csv --yfinance --benchmark URTH --mar 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import AnalysisConfig
from .engine import analyze
from .analytics.recommend import flag_icon
from .report import charts as C
from .report.markdown import render_markdown, write_report


def _today() -> str:
    # avoid importing datetime.now at module import; only used for stamping
    from datetime import date
    return date.today().isoformat()


def build_config(args) -> AnalysisConfig:
    cfg = AnalysisConfig()
    if args.mar is not None:
        cfg.mar = args.mar
    if args.risk_free is not None:
        cfg.risk_free = args.risk_free
    if args.benchmark:
        cfg.benchmark = args.benchmark
    if args.max_crypto is not None:
        cfg.max_crypto_weight = args.max_crypto
    return cfg


def load_data(args, cfg: AnalysisConfig):
    if args.sample:
        from .data.sample import make_sample
        return make_sample()
    if not args.holdings:
        print("error: provide --holdings CSV, or use --sample", file=sys.stderr)
        sys.exit(2)
    if args.yfinance:
        from .data.yfinance_adapter import load_yfinance
        holdings = pd.read_csv(args.holdings)
        holdings.columns = [c.strip().lower() for c in holdings.columns]
        return load_yfinance(
            list(holdings["ticker"]),
            period=args.period,
            benchmark=cfg.benchmark,
            holdings=holdings,
        )
    from .data.csv_adapter import load_csv
    return load_csv(args.holdings, args.prices, args.fundamentals, benchmark=cfg.benchmark)


def print_console(result, cfg):
    m = result.metrics
    print("\n=== Portfolio-Analyse ===")
    if result.meta.get("synthetic"):
        print("  (⚠️ synthetische Sample-Daten)")
    if m:
        print(f"  Ann. Rendite : {m.get('annualized_return', float('nan')):.2%}")
        print(f"  Volatilität  : {m.get('volatility', float('nan')):.2%}")
        print(f"  Sortino (MAR {cfg.mar:.1%}): {m.get('sortino', float('nan')):.2f}   "
              f"Sharpe: {m.get('sharpe', float('nan')):.2f}")
        print(f"  Max Drawdown : {m.get('max_drawdown', float('nan')):.1%}")
        print(f"  VaR/CVaR 95% : {m.get('var_95', float('nan')):.2%} / "
              f"{m.get('cvar_95', float('nan')):.2%}")
    print("\n--- Diagnostik ---")
    for f in result.flags:
        print(f"  {flag_icon(f.level)} [{f.category}] {f.message}")
    if not result.factor_scores.empty:
        print("\n--- Top Faktor-Composite ---")
        for tkr, row in result.factor_scores.head(5).iterrows():
            print(f"  {tkr:8s} {row['composite']:+.2f}")
    print()


def render_reports(result, cfg, report_path: Path):
    assets_dir = report_path.parent / "_analyzer_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    stem = report_path.stem
    charts = {}
    if not result.equity_curve.empty:
        p = assets_dir / f"{stem}-equity.png"
        C.equity_curve_chart(result.equity_curve, result.meta.get("benchmark_curve"), p)
        charts["equity_curve"] = f"_analyzer_assets/{p.name}"
    if not result.drift.empty:
        p = assets_dir / f"{stem}-alloc.png"
        C.allocation_chart(result.allocation, cfg.target_allocation, p)
        charts["allocation"] = f"_analyzer_assets/{p.name}"
    if not result.factor_scores.empty:
        p = assets_dir / f"{stem}-factors.png"
        C.factor_chart(result.factor_scores, p)
        charts["factor"] = f"_analyzer_assets/{p.name}"
    if not result.correlation.empty and result.correlation.shape[0] >= 2:
        p = assets_dir / f"{stem}-corr.png"
        C.correlation_chart(result.correlation, p)
        charts["correlation"] = f"_analyzer_assets/{p.name}"

    md = render_markdown(result, cfg, title=stem.replace("-", " ").title(),
                         charts=charts, created=_today())
    write_report(md, report_path)
    return report_path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="portfolio-analyzer",
                                 description="Evidence-based stock & portfolio diagnostics.")
    ap.add_argument("--holdings", help="Holdings CSV (ticker,value,asset_class,ter,region)")
    ap.add_argument("--prices", help="Prices CSV (wide or long)")
    ap.add_argument("--fundamentals", help="Fundamentals CSV (ticker,pe,pb,roe,...)")
    ap.add_argument("--yfinance", action="store_true", help="Fetch prices+fundamentals via yfinance")
    ap.add_argument("--period", default="5y", help="yfinance history period (default 5y)")
    ap.add_argument("--sample", action="store_true", help="Use built-in synthetic sample data")
    ap.add_argument("--benchmark", default=None, help="Benchmark ticker (e.g. URTH)")
    ap.add_argument("--mar", type=float, default=None, help="Minimum Acceptable Return, annualised")
    ap.add_argument("--risk-free", type=float, default=None, help="Risk-free rate, annualised")
    ap.add_argument("--max-crypto", type=float, default=None, help="Crypto weight alert threshold")
    ap.add_argument("--report", help="Write a Markdown report to this path")
    ap.add_argument("--json", help="Write machine-readable JSON metrics to this path")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    data = load_data(args, cfg)
    result = analyze(data, cfg)

    print_console(result, cfg)

    if args.report:
        path = render_reports(result, cfg, Path(args.report))
        print(f"→ Report: {path}")
    if args.json:
        import json
        payload = {
            "metrics": result.metrics,
            "benchmark_metrics": result.benchmark_metrics,
            "diversification": result.diversification,
            "flags": [f.as_dict() for f in result.flags],
            "factor_scores": result.factor_scores.round(3).to_dict(orient="index"),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"→ JSON: {args.json}")


if __name__ == "__main__":
    main()
