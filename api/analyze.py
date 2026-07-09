"""Vercel Python serverless function: POST market data -> JSON diagnostics.

Reuses the pure `portfolio_analyzer` analytics core (numpy/pandas only — no
matplotlib, charts are rendered client-side). Accepts either the built-in
synthetic sample or user-uploaded CSV text.

Request body (JSON):
{
  "mode": "sample" | "csv" | "tickers",
  "config": {"mar": 0.02, "risk_free": 0.03, "benchmark": "URTH", "max_crypto": 0.05},
  "holdings_csv": "...", "prices_csv": "...", "fundamentals_csv": "...",  # mode=csv
  "holdings": [{"ticker": "AAPL", "weight": 40, "type": "Aktie"}, ...]     # mode=tickers
}

mode=tickers fetches live prices (yfinance -> Stooq fallback) and, always,
per-ticker fundamentals (capped + time-budgeted). All soft failures surface in
meta.warnings rather than raising.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
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


_TYPE_MAP = {
    "etf": ("equity", "etf"), "aktie": ("equity", "stock"), "stock": ("equity", "stock"),
    "krypto": ("crypto", "crypto"), "crypto": ("crypto", "crypto"),
    "anleihe": ("bond", "etf"), "bond": ("bond", "etf"),
    "gold": ("gold", "etf"), "cash": ("cash", "etf"),
}


_ISIN_RE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")


def _is_isin(s: str) -> bool:
    """True if the string is exactly a 12-char ISIN (2 letters + 9 alnum + check digit)."""
    return bool(_ISIN_RE.fullmatch((s or "").strip().upper()))


def _yf_symbol(ticker: str, asset_class: str) -> str:
    """Display ticker -> Yahoo symbol. Crypto without a pair suffix -> '<T>-USD'.

    Empty string signals 'needs ISIN resolution' (filled in later, with network).
    """
    if _is_isin(ticker):
        return ""
    if asset_class == "crypto" and "-" not in ticker:
        return f"{ticker}-USD"
    return ticker


# OpenFIGI exchange code -> Yahoo symbol suffix. US venues take no suffix.
_EXCH_SUFFIX = {
    "US": "", "UN": "", "UW": "", "UQ": "", "UA": "", "UP": "", "UR": "", "UV": "", "PQ": "", "UD": "",
    "GY": ".DE", "GR": ".DE", "GF": ".F", "GM": ".MU", "GS": ".SG", "GD": ".DU", "GB": ".BE",
    "NA": ".AS", "AV": ".VI", "LN": ".L", "FP": ".PA", "IM": ".MI", "SM": ".MC",
    "SW": ".SW", "VX": ".SW", "SE": ".SW", "SS": ".ST", "DC": ".CO", "NO": ".OL", "FH": ".HE",
    "PL": ".LS", "BB": ".BR", "ID": ".IR", "CN": ".TO", "CT": ".TO", "JP": ".T", "JT": ".T",
    "HK": ".HK", "AU": ".AX", "AT": ".AX", "NZ": ".NZ", "SP": ".SI",
}
# preferred listing order when an ISIN maps to several exchanges (US first, then Xetra, then EU)
_PREFERRED_EXCH = ["US", "UW", "UN", "UQ", "UA", "GY", "GR", "NA", "LN", "IM", "SW", "FP", "AV", "SM", "SS"]


def _figi_to_yahoo(ticker, exch):
    """OpenFIGI (ticker, exchCode) -> Yahoo symbol, or None if the exchange is unmapped."""
    if not ticker:
        return None
    suffix = _EXCH_SUFFIX.get((exch or "").upper())
    if suffix is None:
        return None
    return ticker.replace("/", "-").replace(" ", "-") + suffix


def _pick_listing(data):
    """From OpenFIGI's per-exchange listings, pick the most Yahoo-friendly one."""
    best, best_rank = None, 10_000
    for d in data:
        exch = (d.get("exchCode") or "").upper()
        if exch not in _EXCH_SUFFIX:
            continue
        rank = _PREFERRED_EXCH.index(exch) if exch in _PREFERRED_EXCH else 500
        if rank < best_rank:
            best, best_rank = d, rank
    return best


def _resolve_isins_openfigi(isins: list, timeout: float = 8.0) -> dict:
    """Batch-resolve ISINs -> Yahoo symbols via OpenFIGI (datacenter-friendly, keyless).

    Returns {isin: yahoo_symbol} for those that mapped. OPENFIGI_API_KEY (optional)
    raises the rate limit.
    """
    import urllib.request

    isins = list(dict.fromkeys(isins))
    if not isins:
        return {}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OPENFIGI_API_KEY")
    if key:
        headers["X-OPENFIGI-APIKEY"] = key
    out = {}
    for i in range(0, len(isins), 10):  # keyless limit: 10 jobs per request
        chunk = isins[i:i + 10]
        body = json.dumps([{"idType": "ID_ISIN", "idValue": x} for x in chunk]).encode("utf-8")
        try:
            req = urllib.request.Request("https://api.openfigi.com/v3/mapping",
                                         data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                results = json.loads(r.read())
        except Exception:
            continue
        for isin, res in zip(chunk, results):
            data = (res or {}).get("data") or []
            best = _pick_listing(data)
            if best:
                sym = _figi_to_yahoo(best.get("ticker"), best.get("exchCode"))
                if sym:
                    out[isin] = sym
    return out


def _resolve_isin_yf(isin: str):
    """Secondary resolver via yfinance's own Search — shares the session that Yahoo
    price downloads use (so it works wherever yf.download works). None on failure."""
    try:
        import yfinance as yf
        quotes = yf.Search(isin, max_results=5, news_count=0, lists_count=0).quotes or []
    except Exception:
        return None
    for q in quotes:
        if q.get("quoteType") in ("EQUITY", "ETF") and q.get("symbol"):
            return q["symbol"]
    return quotes[0].get("symbol") if quotes and quotes[0].get("symbol") else None


def _holdings_from_rows(rows: list) -> pd.DataFrame:
    """Build a holdings frame from [{ticker, weight, type?, asset_class?, name?}, ...].

    Weights may be percentages or decimals; they are normalised to sum 1. Rows
    with an empty ticker or a non-positive weight are dropped.
    """
    recs = []
    for r in rows:
        tkr = str(r.get("ticker", "")).strip().upper()
        if not tkr:
            continue
        w = float(r.get("weight", r.get("value", 0)) or 0)
        if w <= 0:
            continue
        typ = str(r.get("type", "") or "").strip().lower()
        ac, st = _TYPE_MAP.get(typ, (None, None))
        asset_class = r.get("asset_class") or ac or "equity"
        recs.append({
            "ticker": tkr,
            "value": w,
            "asset_class": asset_class,
            "security_type": r.get("security_type") or st or "etf",
            "name": r.get("name", tkr),
            "isin": tkr if _is_isin(tkr) else (str(r.get("isin", "") or "").strip().upper()),
            "yf_symbol": _yf_symbol(tkr, asset_class),
        })
    if not recs:
        raise ValueError("Keine gültigen Ticker+Gewichte (>0) übergeben.")
    df = pd.DataFrame(recs)
    total = df["value"].sum()
    if total > 0:
        df["value"] = df["value"] / total  # normalise to Σ=1
    return df


def _fetch_prices_yf(symbols: list, period: str = "5y"):
    """yfinance batch download. Returns (wide close df keyed by symbol, error|None)."""
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover - import/runtime env
        return pd.DataFrame(), f"yfinance nicht verfügbar ({type(e).__name__})."
    try:
        raw = yf.download(symbols, period=period, auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:  # network / Yahoo throttling
        return pd.DataFrame(), f"yfinance-Abruf fehlgeschlagen ({type(e).__name__})."
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    return close.dropna(how="all"), None


def _stooq_symbol(sym: str):
    """Best-effort Yahoo->Stooq mapping. Returns None when we won't try (crypto)."""
    s = sym.lower()
    if "-" in s:          # crypto pair like btc-usd — Stooq unreliable, skip
        return None
    if "." in s:          # already carries an exchange suffix (e.g. iwda.as)
        return s
    return f"{s}.us"      # bare ticker -> assume US listing


def _fetch_prices_stooq(symbols: list, budget_s: float = 12.0):
    """Per-symbol Stooq CSV fallback, overall time-budgeted (sequential = timeout risk).

    Returns (wide close df keyed by symbol, [failed]).
    """
    import time
    import urllib.request

    series, failed = {}, []
    start = time.monotonic()
    for sym in symbols:
        s = _stooq_symbol(sym)
        if not s or time.monotonic() - start > budget_s:
            failed.append(sym)
            continue
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        try:
            with urllib.request.urlopen(url, timeout=4) as resp:
                text = resp.read().decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(text))
            if df.empty or "Close" not in df.columns or "Date" not in df.columns:
                failed.append(sym)
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            series[sym] = df.dropna(subset=["Date"]).set_index("Date")["Close"].astype(float)
        except Exception:
            failed.append(sym)
    if not series:
        return pd.DataFrame(), list(symbols)
    return pd.DataFrame(series).sort_index(), failed


def _fetch_fundamentals(holdings: pd.DataFrame, warnings: list,
                        cap: int = 15, budget_s: float = 8.0) -> pd.DataFrame:
    """Best-effort per-ticker fundamentals (non-crypto), capped and time-budgeted.

    Indexed by the DISPLAY ticker so it aligns with prices/holdings. Missing
    fields simply stay NaN — the factor engine skips them.
    """
    import time

    try:
        import yfinance as yf
        from portfolio_analyzer.data.yfinance_adapter import _fundamentals_for
    except Exception:
        return pd.DataFrame()
    targets = holdings[holdings["asset_class"] != "crypto"]
    rows, attempted, skipped_budget = {}, 0, 0
    start = time.monotonic()
    for _, h in targets.iterrows():
        if attempted >= cap:
            warnings.append(f"Fundamentaldaten auf {cap} Titel gedeckelt.")
            break
        if time.monotonic() - start > budget_s:
            skipped_budget += 1
            continue
        try:
            info = _fundamentals_for(yf, h["yf_symbol"])
            if info:
                rows[h["ticker"]] = info
        except Exception:
            pass
        attempted += 1
    if skipped_budget:
        warnings.append(f"Zeitbudget erreicht — Fundamentaldaten für {skipped_budget} "
                        "Titel übersprungen.")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


def _fetch_etf_holdings(symbol: str):
    """yfinance funds_data -> (holdings_list, sectors_dict) for one ETF.

    holdings_list = [{symbol, name, weight}] from top_holdings (weight as a
    fraction). Empty/failed funds -> ([], {}). Yahoo only exposes ~top 10.
    """
    try:
        import yfinance as yf
        fd = yf.Ticker(symbol).funds_data
        top = fd.top_holdings
        sectors = dict(fd.sector_weightings or {})
    except Exception:
        return [], {}
    out = []
    try:
        for sym, row in top.iterrows():
            w = float(row.get("Holding Percent", 0) or 0)
            if w > 1.0:            # some responses give percent, not fraction
                w /= 100.0
            out.append({"symbol": str(sym), "name": str(row.get("Name", sym)), "weight": w})
    except Exception:
        return [], sectors
    # normalise sector weights to fractions too
    sectors = {k: (v / 100.0 if v and v > 1.0 else float(v or 0)) for k, v in sectors.items()}
    return out, sectors


def _aggregate_lookthrough(holdings: pd.DataFrame, per_etf: dict) -> dict:
    """Combine per-ETF top holdings (weighted by portfolio weight) into an
    effective single-name and sector exposure. Direct stocks count as themselves.

    per_etf: {etf_ticker: {"holdings": [...], "sectors": {...}}}. Returns
    {top:[{symbol,name,weight}] desc, sectors:{...}, coverage: disclosed weight}.
    """
    effective, names, sectors = {}, {}, {}
    coverage = 0.0
    for _, h in holdings.iterrows():
        w = float(h["value"])                       # portfolio weight (Σ=1)
        tkr = h["ticker"]
        if tkr in per_etf and per_etf[tkr]["holdings"]:
            disclosed = 0.0
            for u in per_etf[tkr]["holdings"]:
                key = (u.get("symbol") or u.get("name") or "").upper() or u.get("name")
                eff = w * float(u["weight"])
                effective[key] = effective.get(key, 0.0) + eff
                names.setdefault(key, u.get("name") or key)
                disclosed += float(u["weight"])
            coverage += w * min(disclosed, 1.0)
            for s, sw in per_etf[tkr]["sectors"].items():
                sectors[s] = sectors.get(s, 0.0) + w * float(sw)
        elif h["asset_class"] == "equity" and h["security_type"] == "stock":
            key = tkr.upper()
            effective[key] = effective.get(key, 0.0) + w
            names.setdefault(key, h.get("name") or tkr)
            coverage += w
    top = sorted(
        ({"symbol": k, "name": names[k], "weight": v} for k, v in effective.items()),
        key=lambda d: d["weight"], reverse=True,
    )[:30]
    return {"top": top, "sectors": sectors, "coverage": coverage}


_TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,11}")
_NON_TICKER = {"-", "CASH", "USD", "EUR", "GBP", "N/A", "NA", ""}


def _looks_like_ticker(sym: str) -> bool:
    """Filter out cash/placeholder rows in Yahoo top_holdings before pricing."""
    s = (sym or "").strip().upper()
    return s not in _NON_TICKER and bool(_TICKER_RE.fullmatch(s))


def _constituent_analysis(top: list, benchmark: str, period: str, warnings: list, n: int = 20) -> dict:
    """Correlation + per-name risk for the top-N effective single names (one price batch)."""
    from portfolio_analyzer.analytics import returns as R
    from portfolio_analyzer.analytics import risk as Risk
    from portfolio_analyzer.analytics import portfolio as P

    cand = [t for t in top if _looks_like_ticker(t.get("symbol"))][:n]
    if len(cand) < 2:
        return {}
    syms = list(dict.fromkeys(t["symbol"].upper() for t in cand))
    all_syms = syms + ([benchmark] if benchmark else [])
    prices, _err = _fetch_prices_yf(all_syms, period)
    if prices.empty:
        prices, _ = _fetch_prices_stooq(all_syms)
    if prices.empty:
        warnings.append("Einzeltitel-Kurse nicht verfügbar — Korrelation übersprungen.")
        return {}

    bench_ret = None
    if benchmark and benchmark in prices.columns:
        bench_ret = R.to_returns(prices[benchmark])
        prices = prices.drop(columns=[benchmark])
    rets = R.to_returns(prices)
    got = [s for s in syms if s in rets.columns]
    if len(got) < 2:
        warnings.append("Zu wenige Einzeltitel-Kurse für eine Korrelationsmatrix.")
        return {}

    ppy = 252
    meta = {t["symbol"].upper(): t for t in cand}
    per_name = [{
        "symbol": s, "name": meta[s]["name"], "weight": _clean(meta[s]["weight"]),
        "vol": _clean(Risk.volatility(rets[s], ppy)),
        "beta": _clean(Risk.beta(rets[s], bench_ret)) if bench_ret is not None else None,
        "ann_return": _clean(R.annualized_return(rets[s], ppy)),
    } for s in got]
    corr = P.correlation_matrix(rets[got])
    missing = [s for s in syms if s not in got]
    if missing:
        warnings.append(f"Einzeltitel ohne Kurs (Korrelation): {', '.join(missing[:8])}"
                        + (" …" if len(missing) > 8 else ""))
    return {
        "tickers": got,
        "matrix": [[_clean(v) for v in row] for row in corr.values],
        "per_name": per_name,
        "avg_corr": _clean(P.average_pairwise_correlation(rets[got])),
    }


def _market_data_from_payload(payload: dict, cfg: AnalysisConfig) -> MarketData:
    mode = payload.get("mode", "sample")
    if mode == "sample":
        from portfolio_analyzer.data.sample import make_sample
        return make_sample()

    if mode == "tickers":
        rows = payload.get("holdings") or []
        holdings = _holdings_from_rows(rows)
        period = payload.get("period", "5y")
        warnings: list = []

        bench = cfg.benchmark
        if bench and bench.upper() == "WORLD":  # sample-only ticker -> real proxy
            bench = "URTH"
            warnings.append("Benchmark 'WORLD' existiert nur in den Sample-Daten — "
                            "nutze URTH (MSCI World ETF).")

        # resolve ISIN rows (yf_symbol == "") to Yahoo symbols:
        # OpenFIGI batch first, then yfinance-Search per unresolved (bounded).
        pending = list(holdings.index[holdings["yf_symbol"] == ""])
        if pending:
            resolved = _resolve_isins_openfigi([holdings.at[i, "isin"] for i in pending])
            yf_tries = 0
            for idx in pending:
                code = holdings.at[idx, "isin"]
                sym = resolved.get(code)
                if not sym and yf_tries < 10:
                    sym = _resolve_isin_yf(code)
                    yf_tries += 1
                if sym:
                    holdings.at[idx, "yf_symbol"] = sym
                    holdings.at[idx, "ticker"] = sym  # show the resolved symbol
                else:
                    warnings.append(f"ISIN {code} nicht auflösbar (evtl. ein Index oder nicht "
                                    "handelbares Papier) — keine Kurse, zählt aber für die Allokation.")
                    holdings.at[idx, "yf_symbol"] = code  # placeholder, won't match prices

        sym_of = dict(zip(holdings["ticker"], holdings["yf_symbol"]))
        all_syms = list(dict.fromkeys(list(holdings["yf_symbol"]) + ([bench] if bench else [])))

        prices, err = _fetch_prices_yf(all_syms, period)
        source = "yfinance"
        if prices.empty:
            if err:
                warnings.append(err)
            warnings.append("Yahoo lieferte keine Kurse — Stooq-Fallback wird versucht.")
            prices, _failed = _fetch_prices_stooq(all_syms)
            source = "stooq"
            if prices.empty:
                warnings.append("Auch Stooq lieferte keine Kurse — nur Allokations-/"
                                "Konzentrations-Diagnostik ohne Risikometriken.")

        benchmark_prices = None
        if not prices.empty:
            if bench and bench in prices.columns:
                benchmark_prices = prices[bench]
                prices = prices.drop(columns=[bench])
            prices = prices.rename(columns={v: k for k, v in sym_of.items()})
            got = set(prices.columns)
            missing = [t for t in holdings["ticker"] if t not in got]
            if missing:
                warnings.append(f"Keine Kurse für: {', '.join(missing)} "
                                "(Ticker/Börsensuffix prüfen, z.B. .DE/.AS/.VI).")

        fundamentals = _fetch_fundamentals(holdings, warnings)

        # --- ETF look-through: top holdings + sectors per ETF, then aggregate ---
        per_etf, breakdown = {}, []
        if payload.get("lookthrough", True):
            import time
            etfs = holdings[(holdings["security_type"] == "etf")
                            & (holdings["asset_class"].isin(["equity", "bond"]))]
            start, attempted, skipped = time.monotonic(), 0, 0
            for _, h in etfs.iterrows():
                if attempted >= 15:
                    break
                if time.monotonic() - start > 12:
                    skipped += 1
                    continue
                hlds, sectors = _fetch_etf_holdings(h["yf_symbol"])
                attempted += 1
                if hlds:
                    per_etf[h["ticker"]] = {"holdings": hlds, "sectors": sectors}
                    breakdown.append({"ticker": h["ticker"], "name": h.get("name", h["ticker"]),
                                      "holdings": hlds, "sectors": sectors})
            if skipped:
                warnings.append(f"Zeitbudget erreicht — ETF-Durchschau für {skipped} ETF(s) übersprungen.")
        lookthrough = _aggregate_lookthrough(holdings, per_etf) if per_etf or not holdings.empty else {}

        # Top-N constituent correlation + risk (from the effective single names)
        constituents = {}
        if payload.get("lookthrough", True) and lookthrough.get("top"):
            constituents = _constituent_analysis(lookthrough["top"], bench, period, warnings)

        return MarketData(
            prices=prices if not prices.empty else pd.DataFrame(),
            fundamentals=fundamentals, holdings=holdings,
            benchmark_prices=benchmark_prices,
            meta={"source": f"tickers ({source})", "warnings": warnings,
                  "period": period, "etf_breakdown": breakdown, "lookthrough": lookthrough,
                  "constituents": constituents},
        )

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

    lt = result.meta.get("lookthrough") or {}
    lookthrough = {}
    if lt.get("top") or lt.get("sectors"):
        lookthrough = {
            "top": [{"symbol": t["symbol"], "name": t["name"], "weight": _clean(t["weight"])}
                    for t in lt.get("top", [])],
            "sectors": {k: _clean(v) for k, v in (lt.get("sectors") or {}).items()},
            "coverage": _clean(lt.get("coverage")),
        }

    return {
        "meta": {"source": result.meta.get("source"), "synthetic": bool(result.meta.get("synthetic")),
                 "n_obs": m.get("n_obs"), "mar": cfg.mar, "risk_free": cfg.risk_free,
                 "period": result.meta.get("period"),
                 "warnings": [w for w in (result.meta.get("warnings") or []) if w]},
        "metrics": m,
        "benchmark_metrics": bm,
        "diversification": {k: _clean(v) for k, v in result.diversification.items()},
        "allocation_drift": drift,
        "factor_scores": factors,
        "correlation": corr,
        "equity_curve": equity,
        "flags": [f.as_dict() for f in result.flags],
        "etf_breakdown": result.meta.get("etf_breakdown") or [],
        "lookthrough": lookthrough,
        "constituents": result.meta.get("constituents") or {},
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
