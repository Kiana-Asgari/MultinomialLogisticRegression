"""
Multivariate Proximal Oprator
"""

import sys

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import (batched_mlogit,
                                        batched_mlogit_jacobian, batched_mult,
                                        batched_product, batched_sqrtm)
from state_evolution.functions import score_jacobian_batched




def prox_fp_iteration(g_batch, S, max_iter=600, tol=1e-4, verbose=False):
    # computes Prox(g; S) = argmin_beta { g.T S^{-1}g @ mlogit(beta) }
    prox_t = np.zeros_like(g_batch)  # Shape (N, k)
    error = 0
    alpha = 1
    divergence = False


    if np.linalg.norm(S) >  3*1e4:
        prox_t, divergence = prox_newton_iteration(g_batch, S, prox_t, verbose=False)
    else:
        for i in range(max_iter):
            function_batch = g_batch - batched_mult(S, batched_mlogit(prox_t)[:, :-1])
            prox_next = prox_t + alpha * (function_batch - prox_t)
            F_next = batched_mult(S, batched_mlogit(prox_next)[:, :-1]) - g_batch + prox_next
            
            error_new = np.max(np.linalg.norm(F_next, axis=1))
            if error_new > error - 1e-5:
                alpha = max(alpha/2, 2*1e-4)

                # Check for convergence: if the prox_val is small enough, stop
            prox_t = prox_next
            error = error_new

            if error < tol:
                divergence = False
                break
            
 
    
    if error     > 1e-4: pass
        #print(' prox_fp_iteration did not converge with error', error)
        #prox_next, divergence = prox_newton_iteration(g_batch, S, prox_t, verbose=False)
    #print(' prox_fp_iteration converged with error', error, 'iterations: ', i)

    return prox_next, divergence





##########################################################



def best_alpha(F, deriv_inv, prox_t, S, g_batch):
    alpha_values = np.linspace(0.01, 1.2, 20)
    error_values = []
    grad =  np.einsum('nij, ni->nj', deriv_inv, F)

    for i, alpha in enumerate(alpha_values):
        prox_next = prox_t - alpha * grad
        F_alpha = batched_mult(S, batched_mlogit(prox_next)[:, :-1]) - g_batch + prox_next
        error = np.max(np.linalg.norm(F_alpha, axis=1))
        error_values.append(error)

    return alpha_values[np.argmin(error_values)]



def prox_newton_iteration(g_batch, S, prox_fp,\
                           max_iter=10000, tol=1e-2, verbose=False):
    
    divergence = False
   # if verbose:
    print('...trying to compute prox through newton iteration')

    prox_t = -1e1  * np.ones_like(g_batch)  # Shape (N, k)
    N , k = prox_t.shape

    for i in range(max_iter):
        deriv = np.einsum('ij, Njk->Nij',S, batched_mlogit_jacobian(prox_t)) \
                +  np.eye(k)[None, :, :]
        deriv_inv = np.linalg.inv(deriv) # Shape (N, k, k)
        F = batched_mult(S, batched_mlogit(prox_t)[:, :-1]) - g_batch + prox_t # Shape (N, k)

        alpha = best_alpha(F, deriv_inv, prox_t, S, g_batch)
        prox_next = prox_t - alpha * np.einsum('nij, ni->nj', deriv_inv, F)

        error = np.max(np.linalg.norm(F, axis=1))
        if verbose:
            print(f'     **prox_newton_iteration iteration {i} error: {error}, alpha: {alpha}')
        if error < tol:
            break
            
        prox_t = prox_next
    if error > tol:
        divergence = False #changed this
        print(' WARN: prox_newton_iteration did not converge for S = ', S, 'error = ', error)
        print('HALTING...')
    print('    *** prox with newton converged with n-iter: ', i, 'error: ', error)
   
    return prox_next, divergence




def prox_fp_root(g, S):
    prox_initial_guess = np.zeros_like(g)
    prox, infodict, ier, mesg = fsolve(prox_deriv, prox_initial_guess, args=(g, S), full_output=True)
    
    
    print(f" fsolve  status {ier}")
    print(f" Message: {mesg}")
    print(f" Number of function calls: {infodict['nfev']}")
    print(f" Final error: {infodict['fvec']}")
    
    return prox


def prox_deriv(beta, g, S):
    # Compute the proximal derivative
    logit = batched_mlogit(np.array([beta]))[:, :-1].flatten()
    return (beta - g) + S @ logit # Exclude the last element in mlogit



def prox_density(g_0_batch, g_batch, y_batch, A_full, cov_inv, S, k, T=None, mean=None, gaussian_IS_weight=None):
        # T = S @ grad \ell(Z) + Z,  gaussian_point = T - S@Y
        if T is None: 
            gradient_batch = batched_mlogit(g_batch)[:, :-1] # grad \ell(Z)
            gaussian_point_batch = batched_mult(S, gradient_batch) + g_batch - batched_mult(S, y_batch) # S @ grad \ell(Z) + Z - S@Y
        else:
            gaussian_point_batch = T - batched_mult(S, y_batch)

        # mean = E[g|g_0] = R_01 r_00^{-1} @ g_0, tilted_gaussian_point = g - mean
        if mean is None:
            mean = batched_mult(A_full, g_0_batch) # E[g|g_0] = R_01 r_00^{-1} @ g_0
        else:
            mean = mean
        tilted_gaussian_point_batch = gaussian_point_batch - mean

        # gaussian_IS_weight =  (g - mean).T @ cov^{-1} @ (g - mean))
        if gaussian_IS_weight is None:
            gaussian_IS_weight = np.einsum('ni,ij,nj->n', g_batch - mean, cov_inv, g_batch - mean)
        else:
            gaussian_IS_weight = gaussian_IS_weight

        #importance weights is exp(-1/2 * (tilted_gaussian_point_batch - mean).T @ cov^{-1} @ (tilted_gaussian_point_batch - mean))
        # divided by exp(-1/2 * (g_batch - mean).T @ cov^{-1} @ (g_batch - mean)) 
        IS_weight_batch_log = -0.5 * (np.einsum('ni,ij,nj->n', tilted_gaussian_point_batch, cov_inv, tilted_gaussian_point_batch) 
                                      - gaussian_IS_weight)
        IS_weight_batch = np.exp(IS_weight_batch_log)
        # volume_factor_batch = prox_volume_factor(g_batch, S, k)
        return IS_weight_batch #* volume_factor_batch

def prox_volume_factor(v_batch, S, k):
    # returns det(I+Jp(v)_root @ S @ Jp(v)_root), v=prox(g + Sy)
    jacobian_batch = batched_mlogit_jacobian(v_batch)
    score_jacobian_batch = np.einsum('nij,jl->nil', jacobian_batch, S) \
                                + np.eye(k)[None, :, :] 
    determinant_batch = np.linalg.det(score_jacobian_batch) # det(I + S @ Jp(v))
    return determinant_batch