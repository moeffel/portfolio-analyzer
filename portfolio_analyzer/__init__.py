"""
portfolio_analyzer — evidence-based stock & portfolio analytics.

Operationalises the findings from the deep-research note
"Portfoliotheorie für Privatanleger (Österreich)":

- PMPT / downside-risk metrics (Sortino vs a personal MAR) — not just Sharpe
- Factor scoring (value, quality, momentum, size, low-volatility) with the
  documented post-publication-decay caveat baked into the interpretation
- Passive-core philosophy: portfolio *diagnostics* and traffic-light flags,
  never "buy this stock" calls
- Crypto integration caveat (BTC ≈ integrated risk asset, no durable decoupling)
- Austria-aware tax notes (KESt 27.5%, Meldefonds, crypto §27b)

The package is deliberately split so the analytics core has NO network or I/O
dependency and is fully unit-testable offline.
"""

__version__ = "0.1.0"

from .config import AnalysisConfig  # noqa: E402

__all__ = ["AnalysisConfig", "__version__"]
