import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data import fetch_options_chain, clean_chain
from src.surface import compute_surface, interpolate_surface

st.set_page_config(
    page_title="Volatility Surface Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRIC_OPTIONS = {
    "Implied Volatility": ("iv",    "RdYlBu_r"),
    "Delta":              ("delta", "Viridis"),
    "Gamma":              ("gamma", "Viridis"),
    "Vega":               ("vega",  "Viridis"),
}


@st.cache_data(ttl=300)
def load_data(ticker: str) -> pd.DataFrame:
    """Fetch, clean, and compute IV+Greeks. Cached for 5 minutes."""
    raw = fetch_options_chain(ticker)
    cleaned = clean_chain(raw)
    spot = float(cleaned["spot"].iloc[0])
    r    = float(cleaned["r"].iloc[0])
    q    = float(cleaned["q"].iloc[0])
    return compute_surface(cleaned, spot, r, q)


def build_figure(df: pd.DataFrame, z_col: str,
                 option_filter: str, colorscale: str) -> go.Figure:
    if option_filter != "both":
        df = df[df["option_type"] == option_filter]

    if len(df) < 10:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for this selection.",
            showarrow=False, font=dict(size=16), xref="paper", yref="paper",
            x=0.5, y=0.5,
        )
        return fig

    try:
        mg, tg, zg = interpolate_surface(df, z_col)
    except ValueError as e:
        fig = go.Figure()
        fig.add_annotation(text=str(e), showarrow=False, font=dict(size=14),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    z_label = z_col.upper() if z_col == "iv" else z_col.capitalize()

    fig = go.Figure(data=[go.Surface(
        x=mg, y=tg, z=zg,
        colorscale=colorscale,
        colorbar=dict(title=z_label, thickness=18),
        hovertemplate=(
            "Moneyness: %{x:.3f}<br>"
            "T2E (yrs): %{y:.3f}<br>"
            f"{z_label}: " + "%{z:.4f}<extra></extra>"
        ),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Time to Expiry (yrs)",
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.7),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Controls")
ticker = st.sidebar.radio(
    "Ticker",
    options=["^SPX", "SPY"],
    index=0,
    help="SPX = European options (theoretically correct for Black-Scholes). "
         "SPY = American-style; early exercise premium is ignored.",
)
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Volatility Surface Engine")

with st.spinner("Fetching options data and computing implied volatilities…"):
    try:
        df = load_data(ticker)
    except Exception as e:
        st.error(f"**Data fetch failed:** {e}\n\nCheck network connectivity and try again.")
        st.stop()

spot       = float(df["spot"].iloc[0])
fetched_at = df["fetched_at"].iloc[0] if "fetched_at" in df.columns else "unknown"

col1, col2 = st.columns([2, 1])
with col1:
    metric = st.selectbox("Z-Axis Metric", list(METRIC_OPTIONS.keys()))
with col2:
    opt_filter = st.radio("Option Type", ["call", "put", "both"], horizontal=True)

z_col, colorscale = METRIC_OPTIONS[metric]

st.plotly_chart(build_figure(df, z_col, opt_filter, colorscale), use_container_width=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.subheader("Summary Statistics")
display = df if opt_filter == "both" else df[df["option_type"] == opt_filter]

if not display.empty:
    stats = {
        "Metric": ["Spot Price", "Contracts", "Avg IV", "IV Range",
                   "T2E Range", "Risk-free Rate", "Data Timestamp"],
        "Value": [
            f"{spot:,.2f}",
            f"{len(display):,}",
            f"{display['iv'].mean():.1%}",
            f"{display['iv'].min():.1%} – {display['iv'].max():.1%}",
            f"{display['time_to_expiry'].min():.2f} – {display['time_to_expiry'].max():.2f} yrs",
            f"{float(display['r'].iloc[0]):.2%}",
            fetched_at,
        ],
    }
    st.table(pd.DataFrame(stats).set_index("Metric"))
else:
    st.info("No data available for this selection.")
