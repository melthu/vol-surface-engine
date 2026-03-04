import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float, option_type: str) -> float:
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (annualized, e.g. 0.043)
    sigma : volatility (annualized, e.g. 0.20)
    q : continuous dividend yield (annualized)
    option_type : "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        # Return intrinsic value — BS undefined at expiry or zero vol
        if option_type == "call":
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1)
                     - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2)
                     - S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got: {option_type!r}")


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, q: float, option_type: str) -> float:
    """
    Solve for implied volatility via Brent's method.

    Returns np.nan if convergence fails (deep ITM/OTM, illiquid contracts).
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return np.nan

    # If market price is at or below intrinsic, no BS sigma can match it
    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price <= intrinsic:
        return np.nan

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        iv = brentq(objective, a=0.001, b=5.0, xtol=1e-6, maxiter=500)
        return float(iv)
    except (ValueError, RuntimeError):
        return np.nan


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           q: float, option_type: str) -> dict:
    """
    Compute Black-Scholes Greeks.

    Returns dict with keys: delta, gamma, vega, theta, rho
    - vega  : price change per 1% (0.01) move in sigma
    - theta : price change per 1 calendar day
    - rho   : price change per 1% (0.01) move in r
    """
    if T <= 0 or sigma <= 0:
        return {"delta": np.nan, "gamma": np.nan,
                "vega": np.nan, "theta": np.nan, "rho": np.nan}

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)

    # Delta
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    # Gamma (same for calls and puts)
    gamma = (np.exp(-q * T) * n_d1) / (S * sigma * sqrt_T)

    # Vega per 1% vol move
    vega = (S * np.exp(-q * T) * n_d1 * sqrt_T) / 100.0

    # Theta per calendar day
    if option_type == "call":
        theta_annual = (
            -(S * np.exp(-q * T) * n_d1 * sigma) / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
    else:
        theta_annual = (
            -(S * np.exp(-q * T) * n_d1 * sigma) / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
    theta = theta_annual / 365.25

    # Rho per 1% rate move
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta),
        "rho":   float(rho),
    }


if __name__ == "__main__":
    print("=== Sanity Checks ===\n")

    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.043, 0.30, 0.0

    # 1. ATM call price
    call = bs_price(S, K, T, r, sigma, q, "call")
    put  = bs_price(S, K, T, r, sigma, q, "put")
    print(f"1. ATM call price : {call:.4f}  (expect ~13–15)")
    print(f"   ATM put  price : {put:.4f}")

    # 2. Put-call parity
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    print(f"\n2. Put-call parity error : {abs(lhs - rhs):.2e}  (expect < 1e-10)")

    # 3. IV round-trip
    iv = implied_vol(call, S, K, T, r, q, "call")
    print(f"\n3. IV round-trip : {iv:.6f}  (expect 0.300000)")
    print(f"   Error         : {abs(iv - sigma):.2e}  (expect < 1e-5)")

    # 4. Greeks signs
    g = greeks(S, K, T, r, sigma, q, "call")
    print(f"\n4. Call Greeks:")
    for k, v in g.items():
        print(f"   {k:6s} = {v:.6f}")
    print(f"   (delta should be ~0.5–0.6, gamma>0, vega>0, theta<0)")

    # 5. Near-expiry deep OTM — IV should be nan, not a crash
    tiny_price = bs_price(100, 130, 0.001, 0.043, 0.30, 0.0, "call")
    iv_otm = implied_vol(tiny_price, 100, 130, 0.001, 0.043, 0.0, "call")
    print(f"\n5. Deep OTM near-expiry price : {tiny_price:.6f}")
    print(f"   implied_vol result          : {iv_otm}  (expect nan or valid float)")
