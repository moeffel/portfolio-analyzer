"""Chart rendering (matplotlib, Agg backend — no display needed).

Charts are saved as PNG next to the report so they embed in Obsidian.
Palette is colour-blind-safe and works on light/dark note backgrounds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def equity_curve_chart(equity: pd.Series, benchmark: pd.Series | None, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(equity.index, equity.values, color=PALETTE[0], lw=1.6, label="Portfolio")
    if benchmark is not None and not benchmark.empty:
        b = benchmark / benchmark.iloc[0]
        ax.plot(b.index, b.values, color=PALETTE[1], lw=1.2, ls="--", label="Benchmark")
    ax.set_title("Kumulierte Wertentwicklung (Start = 1,0)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def allocation_chart(alloc: pd.Series, target: dict, out: Path) -> Path:
    classes = list(dict.fromkeys(list(alloc.index) + list(target.keys())))
    cur = [float(alloc.get(c, 0.0)) for c in classes]
    tgt = [float(target.get(c, 0.0)) for c in classes]
    x = range(len(classes))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([i - 0.2 for i in x], cur, width=0.4, color=PALETTE[0], label="Aktuell")
    ax.bar([i + 0.2 for i in x], tgt, width=0.4, color=PALETTE[2], label="Ziel")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=0)
    ax.set_ylabel("Anteil")
    ax.set_title("Asset-Allokation: Aktuell vs. Ziel")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def factor_chart(scores: pd.DataFrame, out: Path) -> Path:
    cols = [c for c in ["value", "quality", "momentum", "size", "low_vol"] if c in scores]
    fig, ax = plt.subplots(figsize=(7, max(2.4, 0.5 * len(scores))))
    tickers = list(scores.index)
    ypos = range(len(tickers))
    ax.barh(list(ypos), scores["composite"].values, color=PALETTE[0])
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(tickers)
    ax.invert_yaxis()
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_xlabel("Composite Faktor-Score (z)")
    ax.set_title("Faktor-Composite je Titel")
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def correlation_chart(corr: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="#222")
    ax.set_title("Korrelationsmatrix (Tagesrenditen)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out
