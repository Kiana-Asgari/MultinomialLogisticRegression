"""
Multivariate Proximal Oprator
"""

from multinomial_logistic.utils import batched_mlogit, batched_mult, batched_mlogit_jacobian
import sys
import numpy as np
from scipy.optimize import fsolve, minimize


class ProximalOperatorError(RuntimeError):
    """Exception raised when proximal operator computation fails to converge."""
    pass

def prox_fp_iteration(g_batch, S, max_iter=15000, tol=1e-4, verbose=False):
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
            
 
    
    if error     > 1e-4:
        print(' prox_fp_iteration did not converge with error', error)
        #prox_next, divergence = prox_newton_iteration(g_batch, S, prox_t, verbose=False)
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
    if verbose:
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

