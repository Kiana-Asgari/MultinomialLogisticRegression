import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm

from state_evolution.functions import score_batched
from multinomial_logistic.utils import batched_mlogit, batched_outer, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import coloring_transform, batched_mult
from multinomial_logistic.prox import prox_fp_iteration
from cubature import cubature


def R_01_recursion(S_t, R_00, schur_t, R_01_t, alpha, k, k_0):
    #print('         in R_01_recursion... ')
    A_t = R_01_t @ np.linalg.inv(R_00)
    integrand = integration(_R_01_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    R_01 = R_01_t - alpha * S_t @ integrand 
    return R_01

def schur_recursion(S_t, R_00, schur_t, R_01_t, alpha, k, k_0):
    #print('         in schur_recursion... ')
    A_t = R_01_t @ np.linalg.inv(R_00)
    integrand = integration(_schur_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    schur_root = alpha * S_t @ integrand @ S_t
    return schur_root

#####################################################################################

def _schur_integrand(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0):
    """
    full equation: 1/alpha * S_t E[score(v_t, y_t)@ score(v_t, y_t).T] S_t = (R/R_00)_{t+1}
    integrand: score(v_t, y_t)@ score(v_t, y_t).T | g,g_0 ~ N(0, R_t)
    score(v_t, y _t) = P(v) - y
    v = prox(g + Sy)
    """
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur_t)
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch = prox_fp_iteration(g_batch + batched_mult(S_t, y_batch), S_t) # prox(g + yS; S)
        score_batch = score_batched(prox_g_batch, y_batch) # p(prox(g + yS; S)) - y
        integrand += batched_scalar_mult(batched_outer(score_batch, score_batch), prob_y_batch[:, i]) # score @ score.T   

    integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened

def _R_01_integrand(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0):   
    """
    R_{01, t+1} = R_{01,t} - alpha * E[score(v_t, y_t)@ G_0.T]
    score(v_t, y _t) = P(v) - y
    v = prox(g + Sy)
    """
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur_t)   
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch = prox_fp_iteration(g_batch + batched_mult(S_t, y_batch), S_t) # prox(g + yS; S)
        score_batch = score_batched(prox_g_batch, y_batch) # p(prox(g + yS; S)) - y
        integrand += batched_scalar_mult(batched_outer(score_batch, g_0_batch), prob_y_batch[:, i]) # score @ g_0.T

    integrand = batched_scalar_mult(integrand, pdf) 

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened




def integration(integrand, S_canonical, R_00, schur_t, A_t, alpha, k, k_0):
    #print('     integrating... ')
    fdim = k*k
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(S_canonical, R_00, schur_t, A_t, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-8]*ndim, xmax=[8]*ndim, abserr = 1e-4, relerr=1e-4,
                                  maxEval=100000, norm=2)
    if err.any() > 1e-2:
        print('     **[Warning] integration in the R_recursion error is too large**', err)
    #print('     done integrating')
    return expectations.reshape(k, k)
