# Black-Scholes Volatility Surface Engine

A real-time implied volatility surface and scenario analysis dashboard. Fetches live SPX/SPY options chains, computes implied volatilities and Greeks via Black-Scholes, and renders interactive 3D surfaces and 2D heatmaps.

## Features

**Volatility Surface** — 3D interactive surface built from live options data. Toggle between implied volatility, delta, gamma, and vega across moneyness and time-to-expiry. Filter by calls, puts, or both.

**Scenario Analysis** — Side-by-side call and put heatmaps showing how option value and Greeks shift across spot price (±15%) and volatility (5–80%) scenarios. Includes a summary card with current Black-Scholes price, delta, gamma, and vega at your selected parameters. Override spot, risk-free rate, and dividend yield for hypothetical what-if analysis.

## How It Works

1. **Data ingestion** — Pulls full options chains across all expirations via yfinance. Fetches spot price, risk-free rate (from 10Y Treasury), and dividend yield automatically. Falls back from SPX to SPY if data is sparse.

2. **Quality filtering** — Removes zero-volume, wide-spread, and illiquid contracts. Focuses on near-the-money options (0.8–1.2 moneyness) with expiries under 2 years.

3. **IV computation** — Inverts the Black-Scholes formula for each contract using Brent's method (`scipy.optimize.brentq`). Contracts that fail to converge are dropped gracefully.

4. **Greeks** — Closed-form Black-Scholes partial derivatives: delta, gamma, vega (per 1% vol), theta (per day), rho (per 1% rate).

5. **Surface interpolation** — Scattered IV/Greek data points are interpolated onto a regular grid via `scipy.interpolate.griddata` (cubic with linear fallback).

6. **Scenario heatmaps** — Pure theoretical Black-Scholes evaluation on a 50×50 grid. Works without live data.

## Architecture

```
app.py             Streamlit dashboard (both tabs)
src/pricing.py     Black-Scholes formula, IV solver (brentq), Greeks
src/data.py        yfinance options chain fetch + quality filtering
src/surface.py     IV/Greeks computation loop, griddata interpolation
```

## Quickstart

```bash
git clone https://github.com/melthu/vol-surface-engine.git
cd vol-surface-engine
pip install -r requirements.txt
streamlit run app.py
```

## Known Limitations

- **American vs. European**: SPY options allow early exercise; Black-Scholes assumes European-style. The early exercise premium is negligible for equity index options and is ignored. SPX options are European and theoretically exact.
- **Data freshness**: yfinance returns last-trade prices. Outside market hours, data may be stale — the dashboard shows a timestamp.
- **IV convergence**: Deep ITM/OTM and illiquid contracts often fail to converge. These are dropped; the surface is built from contracts that resolve.
- **Interpolation edges**: `scipy.griddata` can produce artifacts near the boundary of the data's convex hull.

## Stack

Python, NumPy, SciPy, Pandas, Plotly, Streamlit, yfinance
