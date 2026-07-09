# portfolio-analyzer

Ein evidenzbasiertes Analyse-Tool für Aktien & Portfolios, das die Erkenntnisse
aus einer Deep-Research zur Portfoliotheorie operationalisiert. Es
**diagnostiziert und bewertet** — bewusst *keine* „Kauf X"-Rufe, weil die Evidenz
(SPIVA: >70 % aktiver Fonds schlagen den Index über 15 J. nicht) genau dagegen
spricht.

Drei Frontends auf **einem** gemeinsamen, voll unit-getesteten Analytik-Kern:

- **CLI** → schreibt Markdown-Reports mit Charts (z. B. in ein Obsidian-Vault)
- **Streamlit-Dashboard** → interaktiv, lokal
- **Web-App (Vercel)** → statisches Frontend + Python-Serverless-Function

## Was es kann

| Bereich | Umsetzung |
|---|---|
| **PMPT / Downside-Risiko** | Sortino relativ zu *deiner* MAR, Downside-Deviation — nicht nur Sharpe |
| **Risikometriken** | Volatilität, Sharpe, Sortino, Treynor, Beta, Max Drawdown, VaR & **CVaR** |
| **Faktor-Scoring** | Value, Quality, Momentum, Size, Low-Vol als cross-sektionale z-Scores + Composite — mit Post-Publication-Decay-Caveat |
| **Portfolio-Struktur** | Effektive Positionszahl (1/HHI), typ-bewusste Konzentration (Einzelaktie ≠ breiter ETF), Korrelationsmatrix |
| **Allokation** | Abgleich mit Ziel-Allokation, Drift, steuerbewusste Rebalancing-Flags |
| **Krypto** | Exposure-Limit-Flag mit Decoupling-Caveat (BTC = integriertes Risiko-Asset) |
| **Österreich-Steuer** | KESt-, Meldefonds- & Krypto-§27b-Hinweise (informativ) |

## Installation (lokal)

```bash
pip install -r requirements-local.txt     # CLI + Charts + yfinance + Dashboard + Tests
# oder minimal (nur Analytik-Kern): pip install -r requirements.txt
```

> `requirements.txt` ist absichtlich schlank (**nur numpy/pandas**) — das ist die
> Deploy-Datei für die Vercel-Serverless-Function. Für lokale Nutzung mit Charts,
> Live-Daten und Dashboard nimm `requirements-local.txt`.

## Nutzung

### CLI
```bash
# Offline-Demo (synthetische Sample-Daten):
python -m portfolio_analyzer.cli --sample --benchmark WORLD --mar 0.02 --report analyse.md

# Eigenes Portfolio aus CSVs:
python -m portfolio_analyzer.cli --holdings samples/holdings.csv \
    --prices samples/prices.csv --fundamentals samples/fundamentals.csv \
    --benchmark WORLD --mar 0.02 --report out.md --json out.json

# Live via yfinance (auf deinem Rechner — kein API-Key):
python -m portfolio_analyzer.cli --holdings my.csv --yfinance --benchmark URTH --mar 0.02
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

### Web-App (lokal testen)
```bash
npm i -g vercel && vercel dev        # startet Frontend + /api/analyze lokal
```

## Deployment auf Vercel

Das Repo ist ein **zero-config Vercel-Projekt**:

- `index.html` (Repo-Root) → statisches Frontend, ausgeliefert unter `/`
- `api/analyze.py` → Python-Serverless-Function (`POST /api/analyze`)
- `requirements.txt` → schlanke Function-Deps (numpy/pandas)
- `vercel.json` → `maxDuration`/`memory` der Function

```bash
vercel            # Preview-Deploy
vercel --prod     # Production
```
Oder das Repo in Vercel importieren (GitHub-Integration) — kein Build-Command nötig.

> Die Web-App rendert Charts **clientseitig** (Chart.js) und braucht daher weder
> matplotlib noch scipy in der Function.

## Web-App: Eingabe-Modi

| Modus | Daten | Kurse/Fundamentals |
|---|---|---|
| **Ticker + Gewichte** | dynamische Zeilen (**ISIN oder Ticker**, Gewicht %, Typ) oder **Screenshot-Upload** | ISIN → Yahoo-Symbol (Suchendpoint); live via yfinance → Stooq-Fallback; Fundamentals immer (gedeckelt, zeitbudgetiert) |
| **Sample** | eingebautes synthetisches Portfolio | offline |
| **CSV-Upload** | `holdings.csv` (+ optional `prices.csv`, `fundamentals.csv`) | aus den CSVs |

**Bild-Upload** schickt einen verkleinerten Screenshot an `/api/extract`, das per
Claude-Vision (forced Tool-Use) Ticker + Gewichte extrahiert und die editierbare
Tabelle vorbefüllt (human-in-the-loop). Zeilen ohne erkannten Ticker werden rot
markiert und müssen ergänzt werden (ISINs ↦ Yahoo-Ticker sind nicht 1:1).

**ETF-Durchschau (Look-Through):** Für jede ETF-Position werden via
`yfinance funds_data` die Top-Holdings + Sektoren geholt und (a) **pro ETF**
aufklappbar sowie (b) **aggregiert** über alle ETFs dargestellt — effektives
Einzelaktien-Exposure (gewichtet; Direktaktien zählen als sich selbst) + Sektor-Mix.
Yahoo legt nur die **Top-~10** je ETF offen → die Card weist die abgedeckte
**Coverage** transparent aus. **Zeitraum** (1y–max) ist im UI umstellbar und steuert
Kursabruf + alle Kennzahlen.

**Einzeltitel-Durchschau (Top-N):** Aus den effektiven Einzeltiteln werden die **Top-~20**
nach Gewicht bepreist (ein Batch-Download) und eine echte **Korrelationsmatrix** + Risiko
je Titel (Vol, Beta, Rendite) berechnet. Grenze: nur die offengelegten Top-Holdings, nicht
der ganze Korb — eine **vollständige** 500+-Titel-Korrelation bräuchte eine bezahlte
Bulk-Kursdaten-API (auf Vercel-Serverless nicht sinnvoll machbar).

### Umgebungsvariablen (Vercel → Project → Settings → Environment Variables)

| Variable | Zweck | Pflicht |
|---|---|---|
| `ANTHROPIC_API_KEY` | Bild-Extraktion (Claude Vision), ~1–5 Cent/Bild | nur für Bild-Upload |
| `VISION_MODEL` | Modell-Override (Default `claude-haiku-4-5-20251001`) | nein |

Ohne `ANTHROPIC_API_KEY` funktionieren alle anderen Modi normal; der Bild-Upload
zeigt eine klare Fehlermeldung.

### Bekannte Grenzen (Live-Daten)

- **Yahoo drosselt Datacenter-IPs** (Vercel): der Kursabruf kann fehlschlagen →
  automatischer **Stooq-Fallback**; scheitert auch der, gibt es Allokations-/
  Konzentrations-Diagnostik ohne Risikometriken + eine Warnung.
- **ISIN-Auflösung**: **OpenFIGI** (datacenter-freundlich, keyless; optional
  `OPENFIGI_API_KEY`) im Batch, dann **yfinance-Search** pro noch offener ISIN
  (nutzt dieselbe Session wie der Kursabruf). Nicht handelbare ISINs (z. B. ein
  Index wie `US6311011026` = NASDAQ 100) haben keinen Kurs-Ticker und bleiben
  bewusst unauflösbar; die Position zählt weiter für die Allokation.
- **Function-Timeout:** „Fundamentals immer" macht viele Netzwerk-Calls. Bei 504
  in Vercel → Project → Settings → **Functions → Max Duration** auf 60 s anheben.
- **Lambda-Größe:** yfinance-Deps (curl_cffi) liegen bei ~211 MB entpackt (Limit
  250 MB). Bei Build-Größenfehler yfinance weiter runterpinnen oder auf Stooq-only.

## CSV-Format

**holdings.csv** (Pflicht) — `value` *oder* `weight` erforderlich:
```
ticker,value,asset_class,ter,security_type
WORLD,45000,equity,0.0020,etf
AAPL,5000,equity,,stock
BTC,8000,crypto,,crypto
```
`security_type` (`etf`/`stock`/`crypto`) steuert die typ-bewusste Konzentrations-
prüfung: ein breiter ETF darf groß sein, eine Einzelaktie nicht.

**prices.csv** (optional) — wide *oder* long; **fundamentals.csv** (optional) —
beliebige Teilmenge aus `pe,pb,fcf_yield,roe,gross_margin,debt_to_equity,market_cap`.
Vollständige Beispiele in `samples/`.

## Architektur

```
portfolio_analyzer/        Python-Paket
  config.py                AnalysisConfig, TaxConfig (AT-Defaults)
  engine.py                analyze() -> AnalysisResult  (einziger Einstiegspunkt)
  analytics/               REINER Kern, kein I/O — voll unit-getestet
    returns.py risk.py factors.py portfolio.py recommend.py
  data/                    Adapter: base(MarketData) · csv · yfinance · sample
  report/                  charts.py (matplotlib) · markdown.py  (nur CLI)
  cli.py
api/analyze.py             Vercel-Serverless-Function (JSON, nutzt engine)
research.html              Zitierter Research-Report (auf Vercel unter /research.html)
index.html                 Web-Frontend (Chart.js, clientseitige Charts)
dashboard/app.py           Streamlit
tests/                     18 Tests (Risiko gegen Closed-Form, Faktoren, Engine)
samples/                   funktionierende Beispiel-CSVs
vercel.json  requirements.txt  requirements-local.txt
```

> Hinweis: Bewusst **keine** `pyproject.toml` im Root — sie würde Vercels neuen
> „Python-App-Entrypoint"-Modus triggern und das `api/`-Functions-Modell aushebeln.
> Lokal via `requirements-local.txt` installieren; die CLI läuft über
> `python -m portfolio_analyzer.cli`.

## Tests
```bash
pytest -q
```

## Grenzen & Ehrlichkeit
- **Kein Backtester, kein Optimierer.** Diagnostik über *aktuelle* Daten, keine Prognose.
- **Faktor-Scores sind Tilt-Rankings, keine Kaufsignale** — Prämien verlieren
  ~26–58 % nach Publikation (McLean & Pontiff 2016).
- **Steuer-Hinweise sind informativ**, nicht rechtsverbindlich — mit Steuerberater/BMF prüfen.
- Sample-Daten sind **synthetisch** (fixed-seed GBM), keine echten Kurse.
