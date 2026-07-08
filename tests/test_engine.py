"""End-to-end smoke test on the deterministic sample data."""
import numpy as np

from portfolio_analyzer.config import AnalysisConfig
from portfolio_analyzer.data.sample import make_sample
from portfolio_analyzer.engine import analyze


def test_engine_runs_on_sample():
    data = make_sample()
    result = analyze(data, AnalysisConfig())

    # metrics computed
    assert np.isfinite(result.metrics["sortino"])
    assert np.isfinite(result.metrics["max_drawdown"])
    assert result.metrics["n_obs"] > 200

    # weights normalise to 1
    assert abs(result.weights.sum() - 1.0) < 1e-9

    # factor scores present for the 5 equity holdings
    assert not result.factor_scores.empty
    assert "composite" in result.factor_scores

    # diagnostics produced, and the ~9% BTC position trips the crypto alert
    cats = {f.category for f in result.flags}
    assert "risk" in cats and "crypto" in cats
    crypto_flags = [f for f in result.flags if f.category == "crypto"]
    assert any(f.level == "alert" for f in crypto_flags)


def test_reproducible():
    a = analyze(make_sample(seed=7), AnalysisConfig())
    b = analyze(make_sample(seed=7), AnalysisConfig())
    assert a.metrics["sortino"] == b.metrics["sortino"]
