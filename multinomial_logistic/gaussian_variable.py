import numpy as np
from scipy.linalg import sqrtm
from multinomial_logistic.utils import batched_mult

def coloring_transform(Z_vectorized, A, R_00, schur_root, alpha, k, k_0):
    Z_top = Z_vectorized[:,:k] # (N, k)
    Z_bottom = Z_vectorized[:,-k_0:] # (N, k_0)
    g = batched_mult(A , Z_bottom) + batched_mult(schur_root, Z_top)
    g_0 = batched_mult(sqrtm(R_00), Z_bottom)

    return g, g_0