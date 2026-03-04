import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from src.pricing import implied_vol, greeks


def compute_surface(df: pd.DataFrame, spot: float,
                    r: float, q: float) -> pd.DataFrame:
    """
    Compute implied volatility and Greeks for each row in a cleaned chain.

    Parameters
    ----------
    df    : cleaned DataFrame from data.clean_chain()
    spot  : current spot price
    r     : risk-free rate
    q     : dividend yield

    Returns
    -------
    DataFrame with added columns: iv, delta, gamma, vega, theta, rho
    Rows where IV computation fails (NaN) are dropped.
    """
    df = df.copy()

    iv_list = []
    greeks_list = []

    for _, row in df.iterrows():
        iv = implied_vol(
            market_price=row["mid_price"],
            S=spot,
            K=row["strike"],
            T=row["time_to_expiry"],
            r=r,
            q=q,
            option_type=row["option_type"],
        )
        iv_list.append(iv)

        if np.isnan(iv):
            greeks_list.append({"delta": np.nan, "gamma": np.nan,
                                 "vega": np.nan, "theta": np.nan, "rho": np.nan})
        else:
            greeks_list.append(
                greeks(S=spot, K=row["strike"], T=row["time_to_expiry"],
                       r=r, sigma=iv, q=q, option_type=row["option_type"])
            )

    df["iv"] = iv_list
    greeks_df = pd.DataFrame(greeks_list, index=df.index)
    df = pd.concat([df, greeks_df], axis=1)

    df.dropna(subset=["iv"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def interpolate_surface(df: pd.DataFrame, z_col: str,
                        grid_resolution: int = 100) -> tuple:
    """
    Interpolate scattered (moneyness, time_to_expiry, z_col) data onto a
    regular grid suitable for plotly.graph_objects.Surface.

    Uses cubic griddata; falls back to linear if cubic produces >30% NaNs.

    Parameters
    ----------
    df              : enriched DataFrame from compute_surface()
    z_col           : column name to use as the Z axis (e.g. "iv", "delta")
    grid_resolution : number of grid points along each axis

    Returns
    -------
    (moneyness_grid, T_grid, Z_grid) — 2D numpy arrays of shape
    (grid_resolution, grid_resolution)
    """
    x = df["moneyness"].values
    y = df["time_to_expiry"].values
    z = df[z_col].values

    # Drop any residual NaNs in the z column
    mask = ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 4:
        raise ValueError(
            f"Only {len(x)} valid data points for '{z_col}' — cannot interpolate."
        )

    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    moneyness_grid, T_grid = np.meshgrid(xi, yi)

    points = np.column_stack([x, y])

    Z_grid = griddata(points, z, (moneyness_grid, T_grid), method="cubic")

    nan_fraction = np.isnan(Z_grid).sum() / Z_grid.size
    if nan_fraction > 0.30:
        Z_grid = griddata(points, z, (moneyness_grid, T_grid), method="linear")

    return moneyness_grid, T_grid, Z_grid
