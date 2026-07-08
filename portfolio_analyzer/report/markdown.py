"""Render an AnalysisResult into an Obsidian-flavoured Markdown note."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..engine import AnalysisResult
from ..analytics.recommend import flag_icon
from ..analytics.factors import FACTOR_CAVEAT


def _pct(x, dp=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{dp}%}"


def _num(x, dp=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{dp}f}"


def _metrics_table(m: dict, bm: dict | None) -> str:
    rows = [
        ("Ann. Rendite", _pct(m.get("annualized_return")), _pct((bm or {}).get("annualized_return"))),
        ("Volatilität (σ)", _pct(m.get("volatility")), _pct((bm or {}).get("volatility"))),
        ("Downside-Dev.", _pct(m.get("downside_deviation")), _pct((bm or {}).get("downside_deviation"))),
        ("Sharpe", _num(m.get("sharpe")), _num((bm or {}).get("sharpe"))),
        ("**Sortino** (MAR)", f"**{_num(m.get('sortino'))}**", _num((bm or {}).get("sortino"))),
        ("Max Drawdown", _pct(m.get("max_drawdown")), _pct((bm or {}).get("max_drawdown"))),
        ("VaR 95% (tägl.)", _pct(m.get("var_95"), 2), _pct((bm or {}).get("var_95"), 2)),
        ("CVaR 95% (tägl.)", _pct(m.get("cvar_95"), 2), _pct((bm or {}).get("cvar_95"), 2)),
        ("Beta", _num(m.get("beta")), "1.00"),
        ("Treynor", _num(m.get("treynor")), "—"),
    ]
    out = ["| Kennzahl | Portfolio | Benchmark |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def render_markdown(
    result: AnalysisResult,
    cfg: AnalysisConfig,
    *,
    title: str = "Portfolio-Analyse",
    charts: dict | None = None,
    created: str = "",
) -> str:
    charts = charts or {}
    m = result.metrics
    parts: list[str] = []

    # frontmatter
    parts.append("---")
    parts.append(f'title: "{title}"')
    parts.append(f"created: {created}")
    parts.append("type: analysis")
    parts.append("tags:\n  - finance\n  - portfolio\n  - analysis")
    parts.append('related:\n  - "[[Portfoliotheorie fur Privatanleger (Osterreich)]]"\n  - "[[areas/Finance]]"')
    parts.append("---\n")

    parts.append(f"# {title}\n")
    src = result.meta.get("source", "?")
    synth = " ⚠️ SYNTHETISCHE SAMPLE-DATEN" if result.meta.get("synthetic") else ""
    parts.append(f"> Datenquelle: `{src}`{synth} · MAR {cfg.mar:.1%} · risk-free {cfg.risk_free:.1%} · "
                 f"n={m.get('n_obs', 0)} Beobachtungen\n")

    # 1. Diagnostics first — the actionable part
    parts.append("## 1. Diagnostik & Ampeln\n")
    alerts = [f for f in result.flags if f.level == "alert"]
    if alerts:
        parts.append(f"**{len(alerts)} rote Flag(s)** — priorisiert behandeln.\n")
    for f in result.flags:
        line = f"- {flag_icon(f.level)} **[{f.category}]** {f.message}"
        parts.append(line)
        if f.rationale:
            parts.append(f"  - _{f.rationale}_")
    parts.append("")

    # 2. Risk metrics
    parts.append("## 2. Risiko- & risikoadjustierte Kennzahlen\n")
    parts.append(_metrics_table(m, result.benchmark_metrics))
    parts.append("")
    parts.append("> **Sortino** (fett) ist die PMPT-Kennzahl: Rendite über deiner MAR je Einheit "
                 "*Downside*-Risiko — für dich aussagekräftiger als Sharpe.\n")
    if "equity_curve" in charts:
        parts.append(f"![[{charts['equity_curve']}]]\n")

    # 3. Allocation
    if not result.drift.empty:
        parts.append("## 3. Asset-Allokation vs. Ziel\n")
        parts.append("| Klasse | Aktuell | Ziel | Drift |\n|---|---|---|---|")
        for cls, r in result.drift.iterrows():
            parts.append(f"| {cls} | {_pct(r['current'])} | {_pct(r['target'])} | {r['drift']:+.1%} |")
        parts.append("")
        if "allocation" in charts:
            parts.append(f"![[{charts['allocation']}]]\n")

    # 4. Diversification
    if result.diversification:
        d = result.diversification
        parts.append("## 4. Diversifikation\n")
        parts.append(f"- Positionen: **{d.get('n_positions')}**, effektiv (1/HHI): "
                     f"**{_num(d.get('effective_holdings'), 1)}**")
        parts.append(f"- HHI: {_num(d.get('hhi'), 3)} · größte Position: {_pct(d.get('largest_position'))}")
        if "avg_pairwise_corr" in d:
            parts.append(f"- Ø paarweise Korrelation: {_num(d.get('avg_pairwise_corr'))}")
        parts.append("")
        if "correlation" in charts:
            parts.append(f"![[{charts['correlation']}]]\n")

    # 5. Factor scores
    if not result.factor_scores.empty:
        parts.append("## 5. Faktor-Scores (cross-sektional, z-normiert)\n")
        s = result.factor_scores
        cols = [c for c in ["value", "quality", "momentum", "size", "low_vol", "composite"] if c in s]
        parts.append("| Titel | " + " | ".join(c.title() for c in cols) + " |")
        parts.append("|" + "---|" * (len(cols) + 1))
        for tkr, row in s.iterrows():
            parts.append("| " + tkr + " | " + " | ".join(_num(row[c]) for c in cols) + " |")
        parts.append("")
        parts.append(f"> ⚠️ {FACTOR_CAVEAT}\n")
        if "factor" in charts:
            parts.append(f"![[{charts['factor']}]]\n")

    # footer
    parts.append("---\n")
    parts.append("_Erzeugt mit `portfolio-analyzer`. Diagnostik & Scoring, keine Anlageberatung. "
                 "Steuerpunkte mit Steuerberater/BMF verifizieren._")
    return "\n".join(parts)


def write_report(md: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path
