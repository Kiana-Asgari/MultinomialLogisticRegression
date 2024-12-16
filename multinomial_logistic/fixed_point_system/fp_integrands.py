import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from multinomial_logistic.utils import batched_mlogit, batched_normal_basis, batched_outer, batched_mult, batched_scalar_mult
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.utils import batched_wrapper


"""
Calulating the integrands inside the integral of the fixed point system. This is vectorized.
A = R_10 @ R_00^{-1/2}
S = R_11 - R_10 @ R_00^{-1} @ R_01
"""


def batched_fixed_point_integrands(Z_batch, S, A, schur, R_00, lambda_reg, alpha, k, k_0): # Z ~ N(0,I_k)
    N = Z_batch.shape[0]
    schur_root = sqrtm(schur)

    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root, alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)

    prob_y_batch = batched_mlogit(g_0_batch)

    integrand1, integrand2, integrand3 = np.zeros((N, k, k)), np.zeros((N, k, k)), np.zeros((N, k, k_0))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N)
        prox_g_batch = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S) # prox(g + yS; S)
        p_batch = batched_mlogit(prox_g_batch)[:, :-1] # P(p(prox(g + yS; S))

        integrand1 += batched_scalar_mult(batched_outer(p_batch - y_batch, p_batch - y_batch), prob_y_batch[:, i]) # (P(prox(g + yS; S)) - y)(P(prox(g + yS; S)) - y)
        integrand2 += batched_scalar_mult(batched_outer(p_batch - y_batch, g_0_batch), prob_y_batch[:, i]) # (P(prox(g + yS; S)) - y)g_0    
        integrand3 += batched_scalar_mult(batched_outer(p_batch - y_batch, prox_g_batch), prob_y_batch[:, i]) # (P(prox(g + yS; S)) - y)prox(g + yS; S)

    integrand1 = batched_scalar_mult(integrand1, pdf)
    integrand2 = batched_scalar_mult(integrand2, pdf)
    integrand3 = batched_scalar_mult(integrand3, pdf)

    integrands = batched_wrapper(integrand1, integrand2, integrand3, k, k_0)
    return integrands




"""
def batched_fixed_point_integrands(Z_batch, S_cononical, A, R_00, schur_cononical, alpha, k, k_0): # Z ~ N(0,I_k)
    N = Z_batch.shape[0]
    S = S_cononical.T @ S_cononical # S is positive semidefinite
    schur_root = sqrtm(schur_cononical.T @ schur_cononical)

    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root, alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)

    prob_y_batch = batched_mlogit(g_0_batch)

    integrand1, integrand2, integrand3 = np.zeros((N, k, k)), np.zeros((N, k, k)), np.zeros((N, k, k_0))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N)
        prox_g_batch = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S) # prox(g + yS; S)
        p_batch = batched_mlogit(prox_g_batch)[:, :-1] # P(p(prox(g + yS; S))

        integrand1 += batched_scalar_mult(batched_outer(p_batch - y_batch, p_batch - y_batch), prob_y_batch[:, i]) # (P(prox(g + yS; S)) - y)(P(prox(g + yS; S)) - y)
        integrand2 += batched_scalar_mult(batched_outer(p_batch - y_batch, g_0_batch), prob_y_batch[:, i]) # (P(prox(g + yS; S)) - y)g_0    
        integrand3 += batched_scalar_mult(batched_outer(p_batch - y_batch, prox_g_batch), prob_y_batch[:, i]) # (P(prox(g + yS; S)) - y)prox(g + yS; S)

    integrand1 = batched_scalar_mult(integrand1, pdf)
    integrand2 = batched_scalar_mult(integrand2, pdf)
    integrand3 = batched_scalar_mult(integrand3, pdf)

    integrands = batched_wrapper(integrand1, integrand2, integrand3, k, k_0)
    return integrands
"""



