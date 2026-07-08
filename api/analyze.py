"""Vercel Python serverless function: POST market data -> JSON diagnostics.

Reuses the pure `portfolio_analyzer` analytics core (numpy/pandas only — no
matplotlib, charts are rendered client-side). Accepts either the built-in
synthetic sample or user-uploaded CSV text.

Request body (JSON):
{
  "mode": "sample" | "csv",
  "config": {"mar": 0.02, "risk_free": 0.03, "benchmark": "WORLD", "max_crypto": 0.05},
  "holdings_csv": "...", "prices_csv": "...", "fundamentals_csv": "..."   # mode=csv
}
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd

# vendored core lives one level up from /api
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_analyzer.config import AnalysisConfig  # noqa: E402
from portfolio_analyzer.engine import analyze  # noqa: E402
from portfolio_analyzer.data.base import MarketData  # noqa: E402


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _build_config(c: dict) -> AnalysisConfig:
    cfg = AnalysisConfig()
    if c.get("mar") is not None:
        cfg.mar = float(c["mar"])
    if c.get("risk_free") is not None:
        cfg.risk_free = float(c["risk_free"])
    if c.get("benchmark"):
        cfg.benchmark = str(c["benchmark"])
    if c.get("max_crypto") is not None:
        cfg.max_crypto_weight = float(c["max_crypto"])
    return cfg


def _read_prices_text(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    lower = {c.lower() for c in df.columns}
    if {"ticker", "close"}.issubset(lower):
        cols = {c.lower(): c for c in df.columns}
        wide = df.pivot(index=cols.get("date", date_col), columns=cols["ticker"], values=cols["close"])
    else:
        wide = df.set_index(date_col)
    wide = wide.sort_index()
    wide.index.name = "date"
    return wide.astype(float)


def _market_data_from_payload(payload: dict, cfg: AnalysisConfig) -> MarketData:
    mode = payload.get("mode", "sample")
    if mode == "sample":
        from portfolio_analyzer.data.sample import make_sample
        return make_sample()

    holdings_csv = payload.get("holdings_csv")
    if not holdings_csv:
        raise ValueError("mode=csv requires 'holdings_csv'")
    holdings = pd.read_csv(io.StringIO(holdings_csv))
    holdings.columns = [c.strip().lower() for c in holdings.columns]
    if "ticker" not in holdings:
        raise ValueError("holdings CSV needs a 'ticker' column")

    prices = pd.DataFrame()
    benchmark_prices = None
    if payload.get("prices_csv"):
        prices = _read_prices_text(payload["prices_csv"])
        if cfg.benchmark in prices.columns:
            benchmark_prices = prices[cfg.benchmark]
            prices = prices.drop(columns=[cfg.benchmark])

    fundamentals = pd.DataFrame()
    if payload.get("fundamentals_csv"):
        fundamentals = pd.read_csv(io.StringIO(payload["fundamentals_csv"]))
        fundamentals.columns = [c.strip().lower() for c in fundamentals.columns]
        fundamentals = fundamentals.set_index("ticker")

    return MarketData(prices=prices, fundamentals=fundamentals, holdings=holdings,
                      benchmark_prices=benchmark_prices, meta={"source": "csv-upload"})


def _serialize(result, cfg) -> dict:
    m = {k: _clean(v) for k, v in result.metrics.items()}
    bm = ({k: _clean(v) for k, v in result.benchmark_metrics.items()}
          if result.benchmark_metrics else None)

    # equity curve — downsample to keep the payload small
    eq = result.equity_curve
    equity = []
    if not eq.empty:
        step = max(1, len(eq) // 400)
        s = eq.iloc[::step]
        equity = [[str(d.date()), _clean(v)] for d, v in s.items()]

    factors = []
    if not result.factor_scores.empty:
        for tkr, row in result.factor_scores.iterrows():
            factors.append({"ticker": tkr, **{c: _clean(row[c]) for c in result.factor_scores.columns}})

    drift = []
    if not result.drift.empty:
        for cls, row in result.drift.iterrows():
            drift.append({"asset_class": cls, "current": _clean(row["current"]),
                          "target": _clean(row["target"]), "drift": _clean(row["drift"])})

    corr = {}
    if not result.correlation.empty and result.correlation.shape[0] >= 2:
        corr = {"tickers": list(result.correlation.columns),
                "matrix": [[_clean(v) for v in r] for r in result.correlation.values]}

    return {
        "meta": {"source": result.meta.get("source"), "synthetic": bool(result.meta.get("synthetic")),
                 "n_obs": m.get("n_obs"), "mar": cfg.mar, "risk_free": cfg.risk_free},
        "metrics": m,
        "benchmark_metrics": bm,
        "diversification": {k: _clean(v) for k, v in result.diversification.items()},
        "allocation_drift": drift,
        "factor_scores": factors,
        "correlation": corr,
        "equity_curve": equity,
        "flags": [f.as_dict() for f in result.flags],
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        self._send(200, {"ok": True, "service": "portfolio-analyzer", "usage": "POST JSON to this endpoint"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            cfg = _build_config(payload.get("config", {}))
            data = _market_data_from_payload(payload, cfg)
            result = analyze(data, cfg)
            self._send(200, _serialize(result, cfg))
        except Exception as e:  # surface a clean error to the UI
            self._send(400, {"error": type(e).__name__, "message": str(e)})
