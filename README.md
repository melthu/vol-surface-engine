# Volatility Surface Engine

An interactive implied volatility surface dashboard built in Python. Fetches live SPX/SPY options data, computes implied volatilities and Greeks via Black-Scholes, and renders a 3D surface you can explore in the browser.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Data is cached for 5 minutes; use the **Refresh Data** button to force a reload.

## Architecture

```
src/pricing.py   — Black-Scholes formula, IV solver (brentq), Greeks
src/data.py      — yfinance options chain fetch and quality filtering
src/surface.py   — IV/Greeks computation loop, griddata interpolation
app.py           — Streamlit dashboard
```

## Dashboard

The app has two tabs:

### Volatility Surface
- **Z-Axis Metric** — switch between Implied Volatility, Delta, Gamma, or Vega
- **Option Type** — filter to calls, puts, or both
- **Refresh Data** — clears cache and re-fetches live data
- Hover over the surface for exact values at any (moneyness, expiry) point

### Scenario Analysis
A 2D heatmap showing how a single option's value changes across spot price and volatility scenarios — no live data required, pure Black-Scholes.

- **Strike (K)** — defaults to current ATM; adjust to any strike
- **Time to Expiry** — in years (e.g. 0.25 = ~3 months)
- **Heatmap Metric** — Price, Delta, Gamma, Vega, or Theta
- X-axis spans current spot ± 15%; Y-axis spans 5%–80% vol
- Dashed lines mark current spot and nearest ATM market IV (when live data is available)

## Known Limitations

- **American-style options**: SPY options allow early exercise; Black-Scholes assumes European-style. The early exercise premium is small for short-dated equity index options and is ignored here.
- **Stale data outside market hours**: yfinance returns last-trade prices. The dashboard shows a timestamp so you know what you're looking at.
- **IV computation failures**: Deep ITM/OTM and illiquid contracts often fail to converge — this is expected. Those rows are dropped and the surface is built from contracts that do resolve.
- **Interpolation artifacts**: `scipy.griddata` can produce anomalies near the boundary of the data's convex hull. Slight NaN patches or extrapolation artifacts at the surface edges are normal.
- **Data source**: All market data comes from [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance). This is suitable for educational/research use, not production trading.
