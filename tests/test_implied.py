# tests/test_implied.py
import numpy as np
from src.bsm.core import bsm_price, implied_vol

def test_implied_vol_recovery_scalar():
    S, K, T, r, q = 100., 100., 1., 0.05, 0.0
    true_sigma = 0.25
    price = bsm_price(S, K, T, r, q, true_sigma, option='call')
    iv = implied_vol(price, S, K, T, r, q, option='call')
    assert abs(float(iv) - true_sigma) < 1e-4

def test_implied_vol_vector():
    S = np.array([100., 100., 100.])
    K = np.array([90., 100., 110.])
    T = np.array([0.5, 1.0, 2.0])
    r = 0.01; q = 0.0
    true_sigma = np.array([0.2, 0.25, 0.3])
    prices = bsm_price(S, K, T, r, q, true_sigma, option='call')
    iv = implied_vol(prices, S, K, T, r, q, option='call')
    assert np.allclose(iv, true_sigma, atol=1e-4)
