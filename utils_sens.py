# utils_sens.py
import numpy as np
from src.bsm.core import bsm_price, bsm_greeks

def price_sensitivity_sigma(S, K, T, r, q, sigma, option='call', eps=1e-4):
    """Return dPrice/dSigma (vega numeric) for arrays K (broadcastable)."""
    # ensure arrays
    K = np.asarray(K)
    base = bsm_price(S, K, T, r, q, sigma, option)
    bumped = bsm_price(S, K, T, r, q, sigma + eps, option)
    return (bumped - base) / eps

def delta_profile(S, Ks, T, r, q, sigma, option='call'):
    """Return delta across strikes (vector)."""
    greeks = bsm_greeks(S, Ks, T, r, q, sigma)
    # keys in bsm_greeks used by your core.py may be 'delta_call' and 'delta_put'
    if option == 'call':
        return greeks['delta_call']
    else:
        return greeks['delta_put']

def delta_sensitivity_wrt_S(S, Ks, T, r, q, sigma, option='call', eps=None):
    """Return numeric derivative d(delta)/dS across strikes."""
    if eps is None:
        eps = max(1e-6, S * 1e-4)
    Ks = np.asarray(Ks)
    greeks_base = bsm_greeks(S, Ks, T, r, q, sigma)
    base_delta = greeks_base['delta_call'] if option == 'call' else greeks_base['delta_put']
    greeks_bumped = bsm_greeks(S + eps, Ks, T, r, q, sigma)
    bumped_delta = greeks_bumped['delta_call'] if option == 'call' else greeks_bumped['delta_put']
    return (bumped_delta - base_delta) / eps
