"""Streamlit dashboard for portfolio-analyzer.

Run:  streamlit run dashboard/app.py
(from the tools/portfolio-analyzer directory)

Lets you load the synthetic sample, upload CSVs, or fetch live data via
yfinance, tweak the MAR / risk-free / crypto cap, and see metrics, allocation,
factor scores, the correlation heatmap and the traffic-light diagnostics live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make the package importable when run via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_analyzer.config import AnalysisConfig  # noqa: E402
from portfolio_analyzer.engine import analyze  # noqa: E402
from portfolio_analyzer.analytics.recommend import flag_icon  # noqa: E402
from portfolio_analyzer.analytics.factors import FACTOR_CAVEAT  # noqa: E402

st.set_page_config(page_title="Portfolio Analyzer", page_icon="📊", layout="wide")
st.title("📊 Portfolio Analyzer — evidenzbasierte Diagnostik")
st.caption("Diagnostik & Scoring, keine Anlageberatung. Passiv-Kern-Philosophie. "
           "Steuerpunkte mit Steuerberater/BMF verifizieren.")

# ---------------- sidebar controls ----------------
with st.sidebar:
    st.header("Daten")
    source = st.radio("Quelle", ["Sample (synthetisch)", "CSV-Upload", "yfinance (live)"])
    st.header("Parameter")
    mar = st.slider("MAR (Minimum Acceptable Return)", 0.0, 0.10, 0.02, 0.005,
                    help="PMPT-Referenz für Downside-Risiko / Sortino.")
    rf = st.slider("Risk-free", 0.0, 0.06, 0.03, 0.005)
    max_crypto = st.slider("Krypto-Limit", 0.0, 0.20, 0.05, 0.01)
    benchmark = st.text_input("Benchmark-Ticker", "WORLD")


@st.cache_data(show_spinner=False)
def _sample():
    from portfolio_analyzer.data.sample import make_sample
    return make_sample()


def load():
    if source == "Sample (synthetisch)":
        st.info("⚠️ Synthetische Sample-Daten — nur zur Demonstration.")
        return _sample()
    if source == "CSV-Upload":
        h = st.file_uploader("holdings.csv", type="csv")
        p = st.file_uploader("prices.csv (optional)", type="csv")
        f = st.file_uploader("fundamentals.csv (optional)", type="csv")
        if not h:
            st.stop()
        from portfolio_analyzer.data.csv_adapter import load_csv
        import tempfile
        paths = {}
        for key, up in {"h": h, "p": p, "f": f}.items():
            if up is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(up.getvalue()); tmp.close()
                paths[key] = tmp.name
        return load_csv(paths["h"], paths.get("p"), paths.get("f"), benchmark=benchmark)
    # yfinance
    tickers = st.text_input("Ticker (kommagetrennt)", "AAPL,MSFT,SPY,AGG")
    if st.button("Daten laden"):
        from portfolio_analyzer.data.yfinance_adapter import load_yfinance
        syms = [t.strip() for t in tickers.split(",") if t.strip()]
        with st.spinner("yfinance…"):
            return load_yfinance(syms, benchmark=benchmark)
    st.stop()


data = load()
cfg = AnalysisConfig()
cfg.mar, cfg.risk_free, cfg.max_crypto_weight, cfg.benchmark = mar, rf, max_crypto, benchmark
result = analyze(data, cfg)
m = result.metrics

# ---------------- KPI row ----------------
c = st.columns(5)
c[0].metric("Ann. Rendite", f"{m.get('annualized_return', float('nan')):.1%}")
c[1].metric("Sortino", f"{m.get('sortino', float('nan')):.2f}",
            help="PMPT: Rendite über MAR je Einheit Downside-Risiko.")
c[2].metric("Sharpe", f"{m.get('sharpe', float('nan')):.2f}")
c[3].metric("Max Drawdown", f"{m.get('max_drawdown', float('nan')):.1%}")
c[4].metric("CVaR 95% (tägl.)", f"{m.get('cvar_95', float('nan')):.2%}")

# ---------------- diagnostics ----------------
st.subheader("Diagnostik & Ampeln")
for f in result.flags:
    with st.container():
        st.markdown(f"{flag_icon(f.level)} **[{f.category}]** {f.message}")
        if f.rationale:
            st.caption(f.rationale)

left, right = st.columns(2)

with left:
    if not result.drift.empty:
        st.subheader("Allokation vs. Ziel")
        st.dataframe(result.drift.style.format("{:.1%}"))
    if not result.equity_curve.empty:
        st.subheader("Wertentwicklung")
        st.line_chart(result.equity_curve)

with right:
    if not result.factor_scores.empty:
        st.subheader("Faktor-Scores")
        st.dataframe(result.factor_scores.style.format("{:.2f}")
                     .background_gradient(cmap="RdYlGn", subset=["composite"]))
        st.caption("⚠️ " + FACTOR_CAVEAT)
    if not result.correlation.empty and result.correlation.shape[0] >= 2:
        st.subheader("Korrelationsmatrix")
        st.dataframe(result.correlation.style.format("{:.2f}")
                     .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))

with st.expander("Alle Kennzahlen (roh)"):
    st.json({k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in m.items()})
