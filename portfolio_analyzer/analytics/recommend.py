"""Evidence-based diagnostics & traffic-light flags.

Philosophy (from the research note): the passive core is sacrosanct and
stock-picking rarely beats the index (SPIVA: >70% of active funds lag over 15y
in 38/39 categories). So this module produces *diagnostics* — structural and
risk flags with a research rationale — not "buy/sell X" calls.

Each check yields a Flag(level, category, message, rationale). Levels:
"ok" (green), "warn" (amber), "alert" (red), "info" (neutral note).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..config import AnalysisConfig


@dataclass
class Flag:
    level: str          # ok | warn | alert | info
    category: str       # concentration | allocation | crypto | risk | cost | tax | diversification
    message: str
    rationale: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


_ICON = {"ok": "🟢", "warn": "🟡", "alert": "🔴", "info": "🔵"}


def flag_icon(level: str) -> str:
    return _ICON.get(level, "•")


def _is_diversified(security_type: str | None, asset_class: str | None) -> bool:
    """A broad fund/ETF spreads idiosyncratic risk internally; a single stock does not."""
    st = (security_type or "").lower()
    if st in {"etf", "fund", "index", "etc"}:
        return True
    if st in {"stock", "equity_single", "crypto"}:
        return False
    # no explicit type: bonds/gold/cash proxies are usually funds; treat unknown equity as single
    return (asset_class or "").lower() in {"bond", "gold", "cash"}


def check_concentration(weights: pd.Series, holdings: pd.DataFrame, cfg: AnalysisConfig) -> list[Flag]:
    """Type-aware concentration: strict limit for single stocks/crypto, lax for diversified funds."""
    flags = []
    if len(weights) == 0:
        return flags
    meta = holdings.set_index("ticker") if "ticker" in holdings else holdings
    worst_single = None
    for tkr, w in weights.items():
        w = float(w)
        st = meta["security_type"].get(tkr) if "security_type" in meta else None
        ac = meta["asset_class"].get(tkr) if "asset_class" in meta else None
        if _is_diversified(st, ac):
            if w > cfg.max_single_fund:
                flags.append(Flag(
                    "warn", "concentration",
                    f"Fonds/ETF {tkr} = {w:.1%} > {cfg.max_single_fund:.0%} — sehr dominant.",
                    "Ein breiter Fonds streut intern, aber >70% in einem Produkt bündelt "
                    "Emittenten-/Produktrisiko. Ggf. auf zwei Anbieter splitten.",
                ))
        else:
            if w > cfg.max_single_position:
                if worst_single is None or w > worst_single[1]:
                    worst_single = (tkr, w)
    if worst_single:
        flags.append(Flag(
            "alert", "concentration",
            f"Einzeltitel {worst_single[0]} = {worst_single[1]:.1%} > Limit "
            f"{cfg.max_single_position:.0%}.",
            "Einzeltitel-Risiko ist unkompensiertes idiosynkratisches Risiko — "
            "durch Diversifikation eliminierbar (MPT). Einzelpositionen begrenzen.",
        ))
    if not flags:
        flags.append(Flag("ok", "concentration",
                          "Keine problematische Einzeltitel-Konzentration."))
    return flags


def check_diversification(div: dict) -> list[Flag]:
    flags = []
    eff = div.get("effective_holdings")
    n = div.get("n_positions")
    if eff is not None and np.isfinite(eff) and n:
        if eff < max(3, 0.4 * n):
            flags.append(Flag(
                "warn", "diversification",
                f"Effektive Positionszahl {eff:.1f} bei {n} Titeln — konzentriert.",
                "Effektive Holdings = 1/HHI. Deutlich unter der nominalen Zahl → "
                "wenige Positionen dominieren das Risiko.",
            ))
        else:
            flags.append(Flag("ok", "diversification",
                              f"Streuung ok (effektiv {eff:.1f} von {n} Positionen)."))
    corr = div.get("avg_pairwise_corr")
    if corr is not None and np.isfinite(corr) and corr > 0.7:
        flags.append(Flag(
            "warn", "diversification",
            f"Ø paarweise Korrelation {corr:.2f} — hoch.",
            "Hohe Korrelation reduziert den Diversifikationsnutzen; Assets bewegen "
            "sich gemeinsam, v.a. im Stress.",
        ))
    return flags


def check_allocation(drift: pd.DataFrame, cfg: AnalysisConfig) -> list[Flag]:
    flags = []
    for cls, row in drift.iterrows():
        d = float(row["drift"])
        if abs(d) > cfg.rebalance_band:
            direction = "über" if d > 0 else "unter"
            flags.append(Flag(
                "warn", "allocation",
                f"{cls}: {row['current']:.1%} ({direction} Ziel {row['target']:.1%}, "
                f"Drift {d:+.1%}) — Rebalancing prüfen.",
                "Drift jenseits des Toleranzbands. In AT Rebalancing bevorzugt über "
                "Netto-Zuflüsse (Sparplan lenken) statt Verkäufe — Verkauf löst 27,5% KESt aus.",
            ))
    if not flags:
        flags.append(Flag("ok", "allocation", "Alle Assetklassen innerhalb des Toleranzbands."))
    return flags


def check_crypto(alloc_by_class: pd.Series, cfg: AnalysisConfig) -> list[Flag]:
    flags = []
    crypto_w = float(alloc_by_class.get("crypto", 0.0)) if alloc_by_class is not None else 0.0
    if crypto_w > cfg.max_crypto_weight:
        flags.append(Flag(
            "alert", "crypto",
            f"Krypto-Exposure {crypto_w:.1%} > Limit {cfg.max_crypto_weight:.0%}.",
            "Bitcoin ist heute ein integriertes Risiko-Asset (Korrelation zu US-Aktien "
            "bis 0,87 in 2024); dauerhaftes Decoupling ist NICHT belegt. Diversifikations"
            "nutzen nur unter Normalbedingungen — als asymmetrische Satellite-Wette begrenzen.",
        ))
    elif crypto_w > 0:
        flags.append(Flag(
            "info", "crypto",
            f"Krypto {crypto_w:.1%} — innerhalb der 1–5%-Satellite-Spanne.",
            "Nicht als Kern-Diversifikator rechnen (Korrelation steigt im Stress). "
            "Krypto-zu-Krypto-Tausch in AT steuerneutral; Fiat-Verkauf 27,5% (§27b).",
        ))
    return flags


def check_risk(metrics: dict, benchmark_metrics: dict | None, cfg: AnalysisConfig) -> list[Flag]:
    flags = []
    sortino = metrics.get("sortino")
    sharpe = metrics.get("sharpe")
    mdd = metrics.get("max_drawdown")
    if sortino is not None and np.isfinite(sortino):
        level = "ok" if sortino >= 1.0 else ("warn" if sortino >= 0.3 else "alert")
        flags.append(Flag(
            level, "risk",
            f"Sortino {sortino:.2f} (MAR {cfg.mar:.1%}), Sharpe {sharpe:.2f}.",
            "Sortino misst Rendite je Einheit Downside-Risiko relativ zu deiner MAR "
            "(PMPT) — für Anleger aussagekräftiger als Sharpe.",
        ))
    if mdd is not None and np.isfinite(mdd):
        level = "ok" if mdd > -0.2 else ("warn" if mdd > -0.4 else "alert")
        flags.append(Flag(
            level, "risk",
            f"Maximaler Drawdown {mdd:.1%}.",
            "Pfadabhängiges Verlustrisiko — der psychologisch relevanteste 'Schmerz'-Indikator "
            "fürs Durchhalten.",
        ))
    if benchmark_metrics:
        b_sortino = benchmark_metrics.get("sortino")
        if sortino is not None and b_sortino is not None and np.isfinite(b_sortino):
            rel = "über" if sortino >= b_sortino else "unter"
            flags.append(Flag(
                "info", "risk",
                f"Sortino {rel} Benchmark ({sortino:.2f} vs {b_sortino:.2f}).",
                "Aktive Abweichung schlägt selten den Index nach Kosten (SPIVA).",
            ))
    return flags


def check_cost(holdings: pd.DataFrame) -> list[Flag]:
    flags = []
    if "ter" in holdings and "weight" in holdings:
        valid = holdings.dropna(subset=["ter"])
        if len(valid):
            wter = float((valid["ter"] * valid["weight"]).sum() / valid["weight"].sum())
            level = "ok" if wter <= 0.003 else ("warn" if wter <= 0.006 else "alert")
            flags.append(Flag(
                level, "cost",
                f"Gewichtete TER ≈ {wter:.2%} p.a.",
                "TER ist einer der wenigen robusten Prädiktoren relativer Fondsrendite — "
                "je niedriger desto besser. Tracking Difference zusätzlich prüfen.",
            ))
    return flags


def tax_notes(cfg: AnalysisConfig) -> list[Flag]:
    if cfg.tax.jurisdiction != "AT":
        return []
    return [
        Flag("info", "tax",
             "AT-Steuer-Checkliste (mit Steuerberater/BMF verifizieren):",
             "• Nur MELDEFONDS kaufen (OeKB-gemeldet) — Nicht-Meldefonds werden pauschal "
             "strafbesteuert. • KESt 27,5% automatisch bei inländischem Depot. "
             "• Rebalancing steuerbewusst über Zuflüsse. • Krypto §27b: Fiat-Verkauf 27,5%, "
             "Krypto-zu-Krypto steuerneutral, Altvermögen (vor 1.3.2021) steuerfrei."),
    ]


def diagnose(
    *,
    weights: pd.Series,
    alloc_by_class: pd.Series,
    drift: pd.DataFrame,
    diversification: dict,
    metrics: dict,
    benchmark_metrics: dict | None,
    holdings: pd.DataFrame,
    cfg: AnalysisConfig,
) -> list[Flag]:
    """Run all checks and return an ordered flag list (alerts first)."""
    flags: list[Flag] = []
    flags += check_risk(metrics, benchmark_metrics, cfg)
    flags += check_concentration(weights, holdings, cfg)
    flags += check_diversification(diversification)
    flags += check_allocation(drift, cfg)
    flags += check_crypto(alloc_by_class, cfg)
    flags += check_cost(holdings)
    flags += tax_notes(cfg)
    order = {"alert": 0, "warn": 1, "ok": 2, "info": 3}
    return sorted(flags, key=lambda f: order.get(f.level, 9))
