"""Unit tests for risk metrics — checked against closed-form / known values."""
import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.analytics import risk as Risk
from portfolio_analyzer.analytics import returns as R


def test_volatility_annualization():
    # constant daily std -> annualised by sqrt(252)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 100_000))
    vol = Risk.volatility(r, 252)
    assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.02)


def test_downside_deviation_only_counts_shortfalls():
    # all returns above MAR -> zero downside deviation
    r = pd.Series([0.01, 0.02, 0.03, 0.015])
    assert Risk.downside_deviation(r, mar=0.0, periods_per_year=252) == 0.0


def test_downside_deviation_symmetry_case():
    # symmetric around 0 with MAR=0: downside dev uses only negatives / full N
    r = pd.Series([-0.02, 0.02, -0.02, 0.02])
    periodic_mar = 0.0
    expected = np.sqrt(np.mean([min(x - periodic_mar, 0.0) ** 2 for x in r])) * np.sqrt(252)
    assert Risk.downside_deviation(r, 0.0, 252) == pytest.approx(expected)


def test_sortino_higher_when_downside_smaller():
    # two series, same mean, one with milder downside -> higher Sortino
    good = pd.Series([0.01, 0.01, -0.005, 0.012] * 50)
    bad = pd.Series([0.02, 0.02, -0.05, 0.006] * 50)
    s_good = Risk.sortino_ratio(good, 0.0, 252)
    s_bad = Risk.sortino_ratio(bad, 0.0, 252)
    assert s_good > s_bad


def test_max_drawdown_known():
    # price 100 -> 120 -> 60 -> 90 : max DD = -50% (120 to 60)
    prices = pd.Series([100, 120, 60, 90])
    r = R.to_returns(prices)
    assert Risk.max_drawdown(r) == pytest.approx(-0.5)


def test_beta_of_self_is_one():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 500))
    assert Risk.beta(b, b) == pytest.approx(1.0)


def test_beta_scaling():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0, 0.01, 5000))
    a = 2.0 * b  # perfectly correlated, 2x amplitude
    assert Risk.beta(a, b) == pytest.approx(2.0, rel=1e-6)


def test_var_cvar_ordering():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 10000))
    var = Risk.value_at_risk(r, 0.95)
    cvar = Risk.conditional_var(r, 0.95)
    assert cvar <= var < 0  # CVaR is deeper in the tail than VaR


def test_summary_keys():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0003, 0.01, 1000))
    s = Risk.summary(r, mar=0.0, risk_free=0.02, periods_per_year=252)
    for k in ["annualized_return", "volatility", "sortino", "sharpe", "max_drawdown", "var_95"]:
        assert k in s
