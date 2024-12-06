import numpy as np
from scipy.stats import multivariate_normal

from multinomial_logistic.utils import batched_mlogit, batched_normal_basis, batched_outer, batched_mult, batched_scalar_mult
from multinomial_logistic.gaussian_variable import coloring_transform
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.utils import batched_wrapper


"""
Calulating the integrands inside the integral of the fixed point system. This is vectorized.
"""

def batched_fixed_point_integrands(Z_batch  , S, A, R_00, schur_root, alpha, k, k_0): # Z ~ N(0,I_k)
    print("in batched fp integrand")
    N = Z_batch.shape[0]
    g_batch, g_0_batch = coloring_transform(Z_batch, A, R_00, schur_root, alpha, k, k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)

    prox_g_batch = prox_fp_iteration(g_batch, S)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand1, integrand2, integrand3 = np.zeros((N, k, k)), np.zeros((N, k, k)), np.zeros((N, k, k_0))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N)
        p_batch = batched_mlogit(prox_g_batch + batched_mult(S, y_batch))[:, :-1]
        integrand1 += batched_scalar_mult(batched_outer(p_batch - y_batch, p_batch - y_batch), prob_y_batch[:, i])
        integrand2 += batched_scalar_mult(batched_outer(p_batch - y_batch, g_0_batch), prob_y_batch[:, i])
        integrand3 += batched_scalar_mult(batched_outer(p_batch - y_batch, prox_g_batch), prob_y_batch[:, i])

    integrand1 = batched_scalar_mult(integrand1, pdf)
    integrand2 = batched_scalar_mult(integrand2, pdf)
    integrand3 = batched_scalar_mult(integrand3, pdf)

    integrands = batched_wrapper(integrand1, integrand2, integrand3, k, k_0)
    print('batched_fixed_point_integrands done with shape', integrands.shape)
    return integrands