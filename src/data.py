import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date

_FALLBACK_R = 0.043  # 4.3% — approximate 10Y Treasury yield


def fetch_options_chain(ticker: str = "^SPX") -> pd.DataFrame:
    """
    Fetch full options chain across all expirations.

    Tries ^SPX first (European-style); falls back to SPY if chain is sparse
    (<50 contracts) or empty.

    Returns a DataFrame with columns:
        strike, expiration, bid, ask, volume, open_interest,
        option_type, spot, r, q, fetched_at
    """
    if ticker.upper() in ("SPX", "^SPX"):
        tickers_to_try = ["^SPX", "SPY"]
    else:
        tickers_to_try = [ticker]

    for sym in tickers_to_try:
        try:
            df, spot, r, q = _fetch_single(sym)
            if df is not None and len(df) >= 50:
                return df
        except Exception:
            continue

    raise RuntimeError(
        "Could not fetch options data from any source. "
        "Check network connectivity and try again later."
    )


def _fetch_single(sym: str):
    """
    Internal: fetch options chain for one symbol.
    Returns (DataFrame, spot, r, q) or (None, None, None, None).
    """
    tkr = yf.Ticker(sym)
    expirations = tkr.options

    if not expirations:
        return None, None, None, None

    # --- Spot price (three-level fallback) ---
    info = {}
    try:
        info = tkr.info or {}
    except Exception:
        pass

    spot = (info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose"))
    if spot is None:
        try:
            hist = tkr.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            spot = None
    if spot is None:
        return None, None, None, None
    spot = float(spot)

    # --- Dividend yield ---
    q = float(info.get("dividendYield") or 0.0)

    # --- Risk-free rate: try ^TNX (10Y yield in percent), else hardcode ---
    try:
        tnx_hist = yf.Ticker("^TNX").history(period="1d")
        r = float(tnx_hist["Close"].iloc[-1]) / 100.0 if not tnx_hist.empty else _FALLBACK_R
    except Exception:
        r = _FALLBACK_R

    # --- Fetch all expirations ---
    frames = []
    for exp in expirations:
        try:
            chain = tkr.option_chain(exp)
            for opt_type, opt_df in [("call", chain.calls), ("put", chain.puts)]:
                if opt_df.empty:
                    continue
                opt_df = opt_df.copy()
                opt_df["option_type"] = opt_type
                opt_df["expiration"] = exp
                keep = [c for c in ["strike", "expiration", "bid", "ask",
                                    "volume", "openInterest", "option_type"]
                        if c in opt_df.columns]
                frames.append(opt_df[keep])
        except Exception:
            continue

    if not frames:
        return None, None, None, None

    df = pd.concat(frames, ignore_index=True)
    df.rename(columns={"openInterest": "open_interest"}, inplace=True)

    df["spot"] = spot
    df["r"] = r
    df["q"] = q
    df["fetched_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return df, spot, r, q


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters and compute derived columns.

    Filters:
    - volume > 0
    - bid > 0
    - mid_price > 0
    - relative spread (ask - bid) / mid_price <= 0.5
    - 0 < time_to_expiry <= 2.0 years
    - 0.8 <= moneyness (strike / spot) <= 1.2

    Adds columns: mid_price, time_to_expiry, moneyness
    """
    df = df.copy()

    # Coerce — yfinance sometimes returns object dtype for numeric columns
    for col in ["strike", "bid", "ask", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["strike", "bid", "ask", "volume"], inplace=True)

    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    # Quality filters
    df = df[df["volume"] > 0]
    df = df[df["bid"] > 0]
    df = df[df["mid_price"] > 0]
    spread_ratio = (df["ask"] - df["bid"]) / df["mid_price"]
    df = df[spread_ratio <= 0.5]

    # Time to expiry
    today = date.today()
    df["expiration_date"] = pd.to_datetime(df["expiration"]).dt.date
    df["time_to_expiry"] = df["expiration_date"].apply(
        lambda exp: (exp - today).days / 365.25
    )
    df = df[(df["time_to_expiry"] > 0) & (df["time_to_expiry"] <= 2.0)]

    # Moneyness
    df["moneyness"] = df["strike"] / df["spot"]
    df = df[(df["moneyness"] >= 0.8) & (df["moneyness"] <= 1.2)]

    df.reset_index(drop=True, inplace=True)
    return df
