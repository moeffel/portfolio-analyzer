"""Offline tests for the serverless API helpers (no network).

Loads api/analyze.py and api/extract.py by path (they aren't an importable
package) and exercises the pure request/response transforms.
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "api", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = _load("analyze")
extract = _load("extract")


# --- analyze: holdings & symbol mapping ---------------------------------------

def test_holdings_normalized_to_one():
    h = analyze._holdings_from_rows([
        {"ticker": "aapl", "weight": 40, "type": "Aktie"},
        {"ticker": "IWDA.AS", "weight": 60, "type": "ETF"},
    ])
    assert abs(h["value"].sum() - 1.0) < 1e-9
    assert list(h["ticker"]) == ["AAPL", "IWDA.AS"]
    assert h.iloc[0]["security_type"] == "stock" and h.iloc[0]["asset_class"] == "equity"


def test_holdings_drops_empty_and_zero():
    h = analyze._holdings_from_rows([
        {"ticker": "VWCE", "weight": 100, "type": "ETF"},
        {"ticker": "", "weight": 50},
        {"ticker": "ZERO", "weight": 0},
    ])
    assert list(h["ticker"]) == ["VWCE"]


def test_holdings_all_invalid_raises():
    with pytest.raises(ValueError):
        analyze._holdings_from_rows([{"ticker": "", "weight": 0}])


def test_crypto_symbol_mapping():
    h = analyze._holdings_from_rows([{"ticker": "btc", "weight": 10, "type": "Krypto"}])
    assert h.iloc[0]["yf_symbol"] == "BTC-USD"
    # already-suffixed crypto pair is left alone
    assert analyze._yf_symbol("ETH-EUR", "crypto") == "ETH-EUR"
    assert analyze._yf_symbol("AAPL", "equity") == "AAPL"


def test_stooq_symbol_mapping():
    assert analyze._stooq_symbol("AAPL") == "aapl.us"
    assert analyze._stooq_symbol("IWDA.AS") == "iwda.as"
    assert analyze._stooq_symbol("BTC-USD") is None  # crypto -> skip Stooq


def test_is_isin():
    assert analyze._is_isin("IE00B4L5Y983")   # iShares Core MSCI World
    assert analyze._is_isin("US0378331005")   # Apple
    assert not analyze._is_isin("AAPL")
    assert not analyze._is_isin("IE00B4L5Y98")   # too short
    assert not analyze._is_isin("")


def test_holdings_isin_row_pending_resolution():
    h = analyze._holdings_from_rows([
        {"ticker": "IE00B4L5Y983", "weight": 100, "type": "ETF"},
    ])
    row = h.iloc[0]
    assert row["isin"] == "IE00B4L5Y983"
    assert row["yf_symbol"] == ""        # empty => needs ISIN resolution later
    assert row["ticker"] == "IE00B4L5Y983"


def test_figi_to_yahoo_suffixes():
    assert analyze._figi_to_yahoo("NKE", "US") == "NKE"        # US -> no suffix
    assert analyze._figi_to_yahoo("EUNL", "GY") == "EUNL.DE"   # Xetra
    assert analyze._figi_to_yahoo("IWDA", "NA") == "IWDA.AS"   # Amsterdam
    assert analyze._figi_to_yahoo("BRK/B", "US") == "BRK-B"    # share class -> '-'
    assert analyze._figi_to_yahoo("X", "ZZ") is None           # unknown exchange
    assert analyze._figi_to_yahoo(None, "US") is None


def test_pick_listing_prefers_us_then_xetra():
    data = [
        {"ticker": "EUNL", "exchCode": "GY"},
        {"ticker": "NKE", "exchCode": "US"},
        {"ticker": "IWDA", "exchCode": "NA"},
    ]
    assert analyze._pick_listing(data)["exchCode"] == "US"
    data2 = [{"ticker": "EUNL", "exchCode": "GY"}, {"ticker": "IWDA", "exchCode": "NA"}]
    assert analyze._pick_listing(data2)["exchCode"] == "GY"
    assert analyze._pick_listing([{"ticker": "X", "exchCode": "ZZ"}]) is None


# --- extract: vision tool-use parsing -----------------------------------------

def _tool_resp(holdings, note=None):
    inp = {"holdings": holdings}
    if note:
        inp["note"] = note
    return {"content": [{"type": "tool_use", "name": "report_holdings", "input": inp}]}


def test_extract_parses_weights_and_flags_unresolved():
    r = extract._normalize(extract._parse_tool_result(_tool_resp([
        {"ticker": "AAPL", "name": "Apple", "weight": 40, "type": "Aktie"},
        {"ticker": "", "isin": "IE00B4L5Y983", "name": "iShares MSCI World", "weight": 60, "type": "ETF"},
    ])))
    assert r["holdings"][0]["unresolved"] is False
    assert r["holdings"][1]["unresolved"] is True  # no ticker -> user must fix
    assert r["holdings"][1]["isin"] == "IE00B4L5Y983"


def test_extract_derives_weight_from_value():
    r = extract._normalize(extract._parse_tool_result(_tool_resp([
        {"ticker": "VWCE", "name": "All-World", "value": 7500, "type": "ETF"},
        {"ticker": "BTC", "name": "Bitcoin", "value": 2500, "type": "Krypto"},
    ])))
    assert r["holdings"][0]["weight"] == 75.0
    assert r["holdings"][1]["weight"] == 25.0


def test_extract_no_tool_block():
    r = extract._normalize(extract._parse_tool_result({"content": [{"type": "text", "text": "hi"}]}))
    assert r["holdings"] == [] and r["note"]
