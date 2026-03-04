import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data import fetch_options_chain, clean_chain
from src.surface import compute_surface, interpolate_surface
from src.pricing import bs_price, greeks

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

HEATMAP_COLORSCALES = {
    "Price": "RdYlBu_r",
    "Delta": "RdBu",
    "Gamma": "Viridis",
    "Vega":  "Viridis",
    "Theta": "Viridis",
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


def build_surface_figure(df: pd.DataFrame, z_col: str,
                          option_filter: str, colorscale: str) -> go.Figure:
    if option_filter != "both":
        df = df[df["option_type"] == option_filter]

    if len(df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for this selection.",
                           showarrow=False, font=dict(size=16),
                           xref="paper", yref="paper", x=0.5, y=0.5)
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


def build_heatmap_figure(override_spot: float, override_r: float, override_q: float,
                          K: float, T: float, opt_type: str,
                          metric: str, market_iv: float) -> go.Figure:
    """50×50 scenario heatmap. Grid centered on override_spot."""
    spots = np.linspace(override_spot * 0.85, override_spot * 1.15, 50)
    vols  = np.linspace(0.05, 0.80, 50)
    S_grid, VOL_grid = np.meshgrid(spots, vols)

    if metric == "Price":
        Z = np.vectorize(
            lambda s, v: bs_price(s, K, T, override_r, v, override_q, opt_type)
        )(S_grid, VOL_grid)
    else:
        greek_key = metric.lower()
        Z = np.vectorize(
            lambda s, v: greeks(s, K, T, override_r, v, override_q, opt_type)[greek_key]
        )(S_grid, VOL_grid)

    fig = go.Figure(data=go.Heatmap(
        x=spots,
        y=vols,
        z=Z,
        colorscale=HEATMAP_COLORSCALES[metric],
        colorbar=dict(title=metric),
        hovertemplate=(
            "Spot: %{x:,.2f}<br>"
            "Vol: %{y:.1%}<br>"
            f"{metric}: " + "%{z:.4f}<extra></extra>"
        ),
    ))

    fig.add_vline(
        x=override_spot,
        line=dict(color="white", width=2, dash="dash"),
        annotation_text="Spot",
        annotation_position="top",
        annotation_font_color="white",
    )

    if market_iv is not None and 0.05 <= market_iv <= 0.80:
        fig.add_hline(
            y=market_iv,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Mkt IV {market_iv:.1%}",
            annotation_position="right",
            annotation_font_color="white",
        )

    fig.update_layout(
        xaxis_title="Spot Price",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".0%",
        margin=dict(l=0, r=20, t=30, b=0),
        height=520,
    )
    return fig


def get_atm_iv(df: pd.DataFrame, T_target: float, opt_type: str):
    """Return the IV of the nearest-ATM contract closest in time to T_target."""
    subset = df[df["option_type"] == opt_type].copy()
    if subset.empty:
        return None
    subset["t_dist"] = (subset["time_to_expiry"] - T_target).abs()
    subset["m_dist"] = (subset["moneyness"] - 1.0).abs()
    close_t = subset[subset["t_dist"] <= 5 / 365.25]
    if close_t.empty:
        close_t = subset.nsmallest(20, "t_dist")
    best = close_t.nsmallest(1, "m_dist")
    return float(best["iv"].iloc[0]) if not best.empty else None


# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Data fetch ────────────────────────────────────────────────────────────────
st.title("Volatility Surface Engine")

data_error = None
df         = None
spot       = 5000.0 if ticker == "^SPX" else 500.0
r, q       = 0.043, 0.0
fetched_at = "unavailable"

with st.spinner("Fetching options data and computing implied volatilities…"):
    try:
        df         = load_data(ticker)
        spot       = float(df["spot"].iloc[0])
        r          = float(df["r"].iloc[0])
        q          = float(df["q"].iloc[0])
        fetched_at = df["fetched_at"].iloc[0] if "fetched_at" in df.columns else "unknown"
    except Exception as e:
        data_error = str(e)

tab_surface, tab_scenario = st.tabs(["Volatility Surface", "Scenario Analysis"])

# ── Tab 1: Volatility Surface ─────────────────────────────────────────────────
with tab_surface:
    if data_error:
        st.error(f"**Data fetch failed:** {data_error}\n\nCheck network connectivity and try again.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            metric = st.selectbox("Z-Axis Metric", list(METRIC_OPTIONS.keys()))
        with col2:
            opt_filter = st.radio("Option Type", ["call", "put", "both"], horizontal=True)

        z_col, colorscale = METRIC_OPTIONS[metric]
        st.plotly_chart(
            build_surface_figure(df, z_col, opt_filter, colorscale),
            use_container_width=True,
        )

        display = df if opt_filter == "both" else df[df["option_type"] == opt_filter]
        if not display.empty:
            stats_df = pd.DataFrame({
                "Metric": ["Spot Price", "Contracts", "Avg IV", "IV Range",
                           "T2E Range", "Risk-free Rate", "Data Timestamp"],
                "Value": [
                    f"{spot:,.2f}",
                    f"{len(display):,}",
                    f"{display['iv'].mean():.1%}",
                    f"{display['iv'].min():.1%} – {display['iv'].max():.1%}",
                    f"{display['time_to_expiry'].min():.2f} – "
                    f"{display['time_to_expiry'].max():.2f} yrs",
                    f"{r:.2%}",
                    fetched_at,
                ],
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=False)

# ── Tab 2: Scenario Analysis ──────────────────────────────────────────────────
with tab_scenario:
    # Row 1 — option parameters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sc_opt_type = st.radio("Option Type", ["call", "put"], horizontal=True,
                               key="sc_opt_type")
    with c2:
        sc_K = st.number_input("Strike (K)", min_value=1.0,
                               value=float(round(spot / 10) * 10),
                               step=10.0, key="sc_K")
    with c3:
        sc_T = st.number_input("Time to Expiry (yrs)", min_value=0.01, max_value=2.0,
                               value=0.25, step=0.05, key="sc_T")
    with c4:
        sc_metric = st.selectbox("Heatmap Metric",
                                 ["Price", "Delta", "Gamma", "Vega", "Theta"],
                                 key="sc_metric")

    # Row 2 — market overrides (collapsed by default)
    with st.expander("Advanced: Override Market Parameters"):
        ov1, ov2, ov3 = st.columns(3)
        with ov1:
            override_spot = st.number_input(
                "Spot Price Override", min_value=1.0,
                value=float(round(spot)),
                step=10.0, key="ov_spot",
            )
        with ov2:
            override_r = st.number_input(
                "Risk-Free Rate Override", min_value=0.0, max_value=1.0,
                value=float(r), step=0.005, format="%.3f", key="ov_r",
            )
        with ov3:
            override_q = st.number_input(
                "Dividend Yield Override", min_value=0.0, max_value=1.0,
                value=float(q), step=0.005, format="%.3f", key="ov_q",
            )

    market_iv = get_atm_iv(df, sc_T, sc_opt_type) if df is not None else None

    st.plotly_chart(
        build_heatmap_figure(
            override_spot=override_spot, override_r=override_r, override_q=override_q,
            K=sc_K, T=sc_T, opt_type=sc_opt_type,
            metric=sc_metric, market_iv=market_iv,
        ),
        use_container_width=True,
    )

    if market_iv is not None:
        st.caption(
            f"Dashed lines: spot ({override_spot:,.2f}) and "
            f"nearest ATM market IV ({market_iv:.1%}). "
            f"r={override_r:.3f}, q={override_q:.3f}."
        )
    else:
        st.caption(
            f"Dashed line: spot ({override_spot:,.2f}). "
            f"r={override_r:.3f}, q={override_q:.3f}. Market IV unavailable."
        )
