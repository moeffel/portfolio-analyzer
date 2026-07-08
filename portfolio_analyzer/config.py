"""Central configuration for an analysis run.

All investor-specific and jurisdiction-specific knobs live here so the
analytics stay pure functions of (data, config).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Trading days per year — annualisation constant for daily series.
TRADING_DAYS = 252


@dataclass
class TaxConfig:
    """Austrian retail defaults (Stand 2026 — verify with Steuerberater/BMF)."""

    kest_rate: float = 0.275  # KESt on gains & dividends (§27a EStG)
    crypto_rate: float = 0.275  # special rate on crypto Neuvermögen (§27b EStG)
    # Crypto-to-crypto swaps are tax-neutral in AT; only crypto->fiat/goods realises.
    crypto_to_crypto_taxable: bool = False
    jurisdiction: str = "AT"


@dataclass
class AnalysisConfig:
    """Everything that parameterises an analysis.

    Attributes
    ----------
    mar : float
        Minimum Acceptable Return (annualised), the PMPT reference point for
        downside risk / Sortino. Default 0.0 = nominal capital preservation.
        Set to your inflation assumption or goal return to be stricter.
    risk_free : float
        Annualised risk-free rate for Sharpe/Treynor.
    benchmark : str
        Ticker used as the market proxy for beta/Treynor/relative stats.
    target_allocation : dict
        Strategic asset-allocation targets by asset class (fractions summing ~1).
        Used for drift/rebalancing diagnostics. Keys are free-form class labels
        matched against each holding's `asset_class`.
    rebalance_band : float
        Absolute drift (in fraction of total) beyond which a class is flagged.
    max_single_position : float
        Concentration flag threshold for any single holding.
    max_crypto_weight : float
        Crypto exposure above this is flagged (research: BTC = integrated risk
        asset, cap for risk-averse investors). Default 0.05 = 5%.
    factor_weights : dict
        Weights for the composite factor score. Momentum/value/quality/size/lowvol.
    """

    mar: float = 0.0
    risk_free: float = 0.03
    benchmark: str = "URTH"  # MSCI World ETF as a broad proxy
    trading_days: int = TRADING_DAYS

    target_allocation: Dict[str, float] = field(
        default_factory=lambda: {
            "equity": 0.80,   # passive core
            "bond": 0.10,
            "crypto": 0.03,   # asymmetric satellite bet, not a diversifier
            "gold": 0.02,
            "cash": 0.05,
        }
    )
    rebalance_band: float = 0.05
    max_single_position: float = 0.10   # strict limit for a single STOCK/crypto
    max_single_fund: float = 0.70        # lax limit for a diversified fund/ETF core
    max_crypto_weight: float = 0.05

    factor_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "value": 0.25,
            "quality": 0.25,
            "momentum": 0.20,
            "size": 0.15,
            "low_vol": 0.15,
        }
    )

    tax: TaxConfig = field(default_factory=TaxConfig)

    def normalized_factor_weights(self) -> Dict[str, float]:
        total = sum(self.factor_weights.values())
        if total <= 0:
            raise ValueError("factor_weights must sum to a positive number")
        return {k: v / total for k, v in self.factor_weights.items()}
