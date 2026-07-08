"""yfinance adapter — live prices + fundamentals (runs on YOUR machine).

NOTE: this container's egress policy blocks Yahoo, so this path is exercised
on the user's own machine, where outbound HTTPS is open. `yfinance` needs no
API key. Import is lazy so the rest of the package works without it installed.
"""
from __future__ import annotations

import pandas as pd

from .base import MarketData


def _lazy_yf():
    try:
        import yfinance as yf  # noqa
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance not installed. Run `pip install yfinance`, or use the "
            "offline CSV adapter / --sample."
        ) from e
    return yf


def _fundamentals_for(yf, ticker: str) -> dict:
    """Best-effort fundamental snapshot from Ticker.info (fields vary/ may be missing)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:  # pragma: no cover - network dependent
        return {}
    roe = info.get("returnOnEquity")
    gm = info.get("grossMargins")
    d2e = info.get("debtToEquity")
    fcf = info.get("freeCashflow")
    mcap = info.get("marketCap")
    fcf_yield = (fcf / mcap) if (fcf and mcap) else None
    return {
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "fcf_yield": fcf_yield,
        "roe": roe,
        "gross_margin": gm,
        # yfinance reports debtToEquity as a percentage (e.g. 150 = 1.5x)
        "debt_to_equity": (d2e / 100.0) if d2e is not None else None,
        "market_cap": mcap,
        "asset_class": "equity",
        "name": info.get("shortName", ticker),
    }


def load_yfinance(
    tickers: list[str],
    period: str = "5y",
    benchmark: str | None = None,
    holdings: pd.DataFrame | None = None,
    with_fundamentals: bool = True,
) -> MarketData:
    yf = _lazy_yf()
    all_syms = list(dict.fromkeys(tickers + ([benchmark] if benchmark else [])))
    raw = yf.download(all_syms, period=period, auto_adjust=True, progress=False)
    # yf returns a column MultiIndex when multiple fields/tickers are present
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(all_syms[0])
    close = close.dropna(how="all")

    benchmark_prices = None
    if benchmark and benchmark in close.columns:
        benchmark_prices = close[benchmark]
        close = close.drop(columns=[benchmark])

    fundamentals = pd.DataFrame()
    if with_fundamentals:
        rows = {t: _fundamentals_for(yf, t) for t in tickers}
        fundamentals = pd.DataFrame.from_dict(rows, orient="index")
        fundamentals.index.name = "ticker"

    if holdings is None:
        # equal-weight fallback if the user only passed tickers
        holdings = pd.DataFrame({"ticker": tickers, "value": [1.0] * len(tickers)})
        holdings["asset_class"] = "equity"

    return MarketData(
        prices=close,
        fundamentals=fundamentals,
        holdings=holdings,
        benchmark_prices=benchmark_prices,
        meta={"source": "yfinance", "period": period},
    )
