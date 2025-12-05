# src/bsm/core.py
from typing import Union
import numpy as np
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]

def _ensure_arrays(*args):
    return tuple(np.asarray(a) for a in args)

def bsm_price(S: ArrayLike, K: ArrayLike, T: ArrayLike,
              r: ArrayLike, q: ArrayLike, sigma: ArrayLike,
              option: str = "call"):
    """
    Vectorized Black-Scholes-Merton price for European call/put.
    S, K, T, r, q, sigma can be scalars or numpy arrays (broadcastable).
    """
    S, K, T, r, q, sigma = _ensure_arrays(S, K, T, r, q, sigma)

    # numerical safety
    sigma = np.where(sigma <= 0, 1e-12, sigma)
    T = np.where(T <= 0, 1e-12, T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option.lower() == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option.lower() == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'")

def bsm_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike,
               r: ArrayLike, q: ArrayLike, sigma: ArrayLike):
    """
    Return dict of delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put.
    Vectorized.
    """
    S, K, T, r, q, sigma = _ensure_arrays(S, K, T, r, q, sigma)
    sigma = np.where(sigma <= 0, 1e-12, sigma)
    T = np.where(T <= 0, 1e-12, T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    delta_call = disc_q * norm.cdf(d1)
    delta_put = delta_call - disc_q
    gamma = (pdf_d1 * disc_q) / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)      # per 1 vol
    theta_call = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                  - r * K * disc_r * norm.cdf(d2)
                  + q * S * disc_q * norm.cdf(d1))
    theta_put = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
    rho_call = K * T * disc_r * norm.cdf(d2)
    rho_put = -K * T * disc_r * norm.cdf(-d2)

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }


# ---------------------------
# Implied volatility solver
# ---------------------------
import math
from typing import Optional

def implied_vol(mkt_price: ArrayLike,
                S: ArrayLike,
                K: ArrayLike,
                T: ArrayLike,
                r: ArrayLike,
                q: ArrayLike,
                option: str = "call",
                initial_guess: float = 0.2,
                tol: float = 1e-8,
                maxiter: int = 100) -> ArrayLike:
    """
    Return implied volatility that matches market option price.
    Strategy: vectorized Newton-Raphson using analytic Vega. If Newton fails
    to converge or produces invalid sigma, fallback to robust bisection on [1e-12, 5].
    Works with scalars and numpy arrays (broadcastable).
    """
    # prepare arrays
    mkt_price, S, K, T, r, q = _ensure_arrays(mkt_price, S, K, T, r, q)
    # initial guess (broadcast)
    sigma = np.full_like(S, float(initial_guess), dtype=float)

    # helper functions (operate on numpy arrays)
    def _price(s):
        return bsm_price(S, K, T, r, q, s, option=option)

    def _vega(s):
        # vega = S * e^{-qT} * pdf(d1) * sqrt(T)
        s = np.where(s <= 0, 1e-12, s)
        d1 = (np.log(S / K) + (r - q + 0.5 * s**2) * T) / (s * np.sqrt(T))
        pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
        return S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    # Newton-Raphson loop (vectorized)
    sigma = np.maximum(sigma, 1e-12)
    for i in range(maxiter):
        price = _price(sigma)
        diff = price - mkt_price
        # check convergence
        if np.all(np.abs(diff) < tol):
            return sigma
        vega = _vega(sigma)
        # avoid division by zero
        step = np.where(vega > 1e-12, diff / vega, np.nan)
        new_sigma = sigma - step
        # guard: keep within (1e-12, 5)
        bad = (new_sigma <= 1e-12) | (new_sigma > 5) | np.isnan(new_sigma)
        sigma = np.where(~bad, new_sigma, sigma)

    # If we reach here, Newton didn't converge for all elements — use bisection elementwise
    lower = np.full_like(sigma, 1e-12)
    upper = np.full_like(sigma, 5.0)

    # ensure that price(lower) <= mkt_price <= price(upper) for a valid bracket
    p_lower = _price(lower)
    p_upper = _price(upper)

    # For any elements where market price is outside [p_lower, p_upper], clip to nearest bound's vol
    out_of_bounds_low = mkt_price <= p_lower
    out_of_bounds_high = mkt_price >= p_upper

    result = np.copy(sigma)

    # elementwise bisection for the remaining indices
    to_solve = ~(out_of_bounds_low | out_of_bounds_high)
    if np.any(out_of_bounds_low):
        result[out_of_bounds_low] = lower[out_of_bounds_low]
    if np.any(out_of_bounds_high):
        result[out_of_bounds_high] = upper[out_of_bounds_high]

    if np.any(to_solve):
        lo = lower[to_solve].astype(float)
        hi = upper[to_solve].astype(float)
        mkt = mkt_price[to_solve]
        S_sub = S[to_solve]; K_sub = K[to_solve]; T_sub = T[to_solve]
        r_sub = r[to_solve]; q_sub = q[to_solve]

        # simple bisection loop
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            # compute price at mid using elementwise call to bsm_price (it broadcasts)
            pmid = bsm_price(S_sub, K_sub, T_sub, r_sub, q_sub, mid, option=option)
            # update bounds
            lo = np.where(pmid > mkt, lo, mid)
            hi = np.where(pmid > mkt, mid, hi)
            if np.all(np.abs(pmid - mkt) < tol):
                break
        result[to_solve] = 0.5 * (lo + hi)

    return result
