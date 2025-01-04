"""
Multivariate Proximal Oprator
First approach: calculated through KKT
"""

from multinomial_logistic.utils import batched_mlogit, batched_mult
import sys
import numpy as np
from scipy.optimize import fsolve, minimize

"""
Fixed point iteration for the proximal operator
vectorized version for a batch of inputs
"""



def prox_fp_iteration(g_batch, S, max_iter=100, tol=1e-5):
    # computes Prox(g; S) = argmin_beta { g.T S^{-1}g @ mlogit(beta) }
    prox_t = np.ones_like(g_batch)  # Shape (N, k)
    flag_converged = False

    for i in range(max_iter):

        prox_next = g_batch - batched_mult(S, batched_mlogit(prox_t)[:, :-1])

        error = np.max(np.linalg.norm(prox_next - prox_t, axis=1))
        bad_indices = np.argmax(np.linalg.norm(prox_next - prox_t, axis=1))
        #print('prox_tt: ', prox_next)
        # Check for convergence: if the prox_val is small enough, stop
        if error < tol:
            flag_converged = True
            break
            
        prox_t = prox_next

    #if not flag_converged:
        #print(' WARN: prox_fp_iteration did not converge for S = ', S, ' and g = ', g_batch[bad_indices], 'error: ', error, 'max_iter: ', i)
        #sys.exit()
    return prox_next






"""
def prox_fp_root(g, S):
    prox_initial_guess = np.zeros_like(g)
    prox = fsolve(prox_deriv, prox_initial_guess, args=(g, S))
    return prox


def prox_deriv(beta, g, S):
    # Compute the proximal derivative
    return (beta - g) + S @ mlogit(beta)  # Exclude the last element in mlogit
"""