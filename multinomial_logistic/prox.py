"""
Multivariate Proximal Oprator
First approach: calculated through KKT
"""

from multinomial_logistic.utils import batched_mlogit, batched_mult

import numpy as np
from scipy.optimize import fsolve, minimize

"""
Fixed point iteration for the proximal operator
vectorized version for a batch of inputs
"""


def prox_deriv_fp(beta_batch, g_batch, S):
    # Compute the proximal derivative as a fixed point equation for each beta in the batch
    return g_batch - batched_mult(S, batched_mlogit(beta_batch)[:, :-1])

def prox_fp_iteration(g_batch, S, max_iter=1000, tol=1e-3):
    beta_batch = np.zeros_like(g_batch)  # Shape (N, k)
    
    for i in range(max_iter):
        fp = prox_deriv_fp(beta_batch, g_batch, S)
        # Check for convergence: if the prox_val is small enough, stop
        if np.linalg.norm(beta_batch - fp) < tol:
            break

        beta_batch = fp
    
    return beta_batch

"""
def prox_fp_root(g, S):
    prox_initial_guess = np.zeros_like(g)
    prox = fsolve(prox_deriv, prox_initial_guess, args=(g, S))
    return prox


def prox_deriv(beta, g, S):
    # Compute the proximal derivative
    return (beta - g) + S @ mlogit(beta)  # Exclude the last element in mlogit
"""