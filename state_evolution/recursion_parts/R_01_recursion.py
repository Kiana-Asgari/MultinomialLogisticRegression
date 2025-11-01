import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

import time

from multinomial_logistic.integration import batched_mult, coloring_transform
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.utils import (batched_mlogit, batched_mult,
                                        batched_normal_basis, batched_outer,
                                        batched_scalar_mult, wrapper)

from state_evolution.utils import integration
from state_evolution.functions import score_batched, score_jacobian_batched
from multinomial_logistic.prox import (prox_density,
                                       prox_fp_iteration, prox_volume_factor)
import time

def R_01_recursion(S_t, S_next, R_00, schur_t, R_01_t, lambda_reg, alpha, k, k_0, R_00_sqrtm_inv=None):
    if R_00_sqrtm_inv is None:
        A_t = R_01_t @ np.linalg.inv(sqrtm(R_00))
    else:
        A_t = R_01_t @ R_00_sqrtm_inv
    start_time = time.time()
    integrand = integration(_R_01_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    end_time = time.time()
    print('  --time without density: ', end_time - start_time, 'value: ', integrand)
    start_time = time.time()
    integrand_with_prox_density = integration(_R_01_integrand_with_prox_density, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    end_time = time.time()
    print('  --time prox density: ', end_time - start_time, 'value: ', integrand_with_prox_density)
    R_01 =  (np.eye(k) - alpha*2*lambda_reg * S_next) @ R_01_t \
            - alpha * S_next @ integrand 

    return R_01





def _R_01_integrand(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0, monte_carlo=False):   
    """
    R_{01, t+1} = R_{01,t} - alpha * E[score(v_t, y_t)@ G_0.T]
    score(v_t, y _t) = P(v) - y
    v = prox(g + Sy)
    """
    global div_prox 
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur_t)   

    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch, div_prox = prox_fp_iteration(g_batch + batched_mult(S_t, y_batch), S_t) # prox(g + yS; S)
        score_batch = score_batched(prox_g_batch, y_batch) # p(prox(g + yS; S)) - y
        integrand += batched_scalar_mult(batched_outer(score_batch, g_0_batch), prob_y_batch[:, i]) # score @ g_0.T

    if not monte_carlo:
        integrand = batched_scalar_mult(integrand, pdf) 

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened



def _R_01_integrand_with_prox_density(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0):
    N = Z_batch.shape[0]
    schur_root = sqrtm(schur_t)
    A_full = A_t @ np.linalg.inv(sqrtm(R_00)) # A_full = A_t @ R_00^{-1/2} = R_01 @ R_00^{-1}
    cov_inv = np.linalg.inv(schur_t)
    global div_prox 
    # Coloring transform to get (g_0) ~ N(0, R_00) from Z_0 ~ N(0, I)
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root, alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ 
    pdf = multivariate_normal(mean=np.zeros(k_0+k), cov=np.eye(k_0+k)).pdf(Z_batch) # density of whitened g_0 only
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))
    score_jacobian_batch = score_jacobian_batched(g_batch, S_t, k) # (I + S @ Jp(v)^{-1}
    det_score_jacobian_batch = 1/np.linalg.det(score_jacobian_batch) # det(I + S @ Jp(v))^{-1}

    gradient_batch = batched_mlogit(g_batch)[:, :-1] # grad \ell(Z)
    T_prox_batch = batched_mult(S_t, gradient_batch) + g_batch
    mean_prox_batch = batched_mult(A_full, g_0_batch)
    gaussian_IS_weight_batch = np.einsum('ni,ij,nj->n', g_batch - mean_prox_batch, cov_inv, g_batch - mean_prox_batch)

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_density_batch = prox_density(g_0_batch=g_0_batch,
                                            g_batch=g_batch,
                                            y_batch=y_batch,
                                            A_full=A_full, 
                                            cov_inv=cov_inv,
                                            S=S_t, 
                                            T=T_prox_batch,
                                            mean=mean_prox_batch,
                                            gaussian_IS_weight=gaussian_IS_weight_batch,
                                            k=k) # density(v|g_0)
        score_batch = score_batched(g_batch, y_batch) # p(g) - y
        firts_term_batch = batched_scalar_mult(batched_outer(score_batch, g_0_batch), prob_y_batch[:, i]) # score @ g_0.T
        second_term_batch = np.einsum('nij,n,n->nij', firts_term_batch, det_score_jacobian_batch, prox_density_batch) # (I + S @ Jp(g))^{-1} * p(y) * density(v,g_0)/density(g|g_0)
        integrand += second_term_batch
   

    integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(V|y))^{-1} * p(y) * p(z_0) * p_prox(v|z_0)
    
    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened