"""Tests for factor scoring and portfolio structure metrics."""
import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.analytics import factors as F
from portfolio_analyzer.analytics import portfolio as P


def test_value_factor_ranks_cheap_higher():
    fund = pd.DataFrame(
        {"pe": [8, 12, 30], "pb": [1.0, 1.5, 6.0]},
        index=["CHEAP", "MID", "EXPENSIVE"],
    )
    scores = F.factor_scores(fund, weights={"value": 1.0})
    assert scores.loc["CHEAP", "value"] > scores.loc["EXPENSIVE", "value"]


def test_quality_penalizes_leverage():
    fund = pd.DataFrame(
        {"roe": [0.25, 0.25], "gross_margin": [0.5, 0.5], "debt_to_equity": [0.2, 3.0]},
        index=["LOWDEBT", "HIGHDEBT"],
    )
    scores = F.factor_scores(fund, weights={"quality": 1.0})
    assert scores.loc["LOWDEBT", "quality"] > scores.loc["HIGHDEBT", "quality"]


def test_size_favors_small_cap():
    fund = pd.DataFrame({"market_cap": [1e9, 1e12]}, index=["SMALL", "MEGA"])
    scores = F.factor_scores(fund, weights={"size": 1.0})
    assert scores.loc["SMALL", "size"] > scores.loc["MEGA", "size"]


def test_momentum_from_prices():
    dates = pd.bdate_range("2020-01-01", periods=300)
    up = pd.Series(np.linspace(100, 200, 300), index=dates)
    down = pd.Series(np.linspace(200, 120, 300), index=dates)
    prices = pd.DataFrame({"UP": up, "DOWN": down})
    fund = pd.DataFrame(index=["UP", "DOWN"])
    scores = F.factor_scores(fund, prices=prices, weights={"momentum": 1.0})
    assert scores.loc["UP", "momentum"] > scores.loc["DOWN", "momentum"]


def test_effective_holdings_equal_weight():
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=list("ABCD"))
    assert P.effective_holdings(w) == pytest.approx(4.0)


def test_effective_holdings_concentrated():
    w = pd.Series([0.97, 0.01, 0.01, 0.01], index=list("ABCD"))
    assert P.effective_holdings(w) < 1.2  # dominated by one position


def test_allocation_drift():
    cur = pd.Series({"equity": 0.9, "bond": 0.1})
    drift = P.allocation_drift(cur, {"equity": 0.8, "bond": 0.1, "crypto": 0.1})
    assert drift.loc["equity", "drift"] == pytest.approx(0.1)
    assert drift.loc["crypto", "drift"] == pytest.approx(-0.1)
