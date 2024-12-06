import numpy as np
from scipy.linalg import sqrtm

def coloring_transform(Z_vectorized, A, R_00, schur_root, alpha, k, k_0):
    Z_top = Z_vectorized[:k] # (k, N)
    Z_bottom = Z_vectorized[-k_0:] # (k_0, N)
    g = A @ Z_bottom + schur_root @ Z_top
    g_0 = sqrtm(R_00) @ Z_bottom

    return g.T, g_0.T