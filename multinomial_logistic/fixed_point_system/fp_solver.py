"""
Solving the fixed point system.
"""
import numpy as np
from scipy.optimize import fsolve
from multinomial_logistic.utils import wrapper, unwrap
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system
from multinomial_logistic.integration import schur_decomposition, schur_complement


def fixed_point_solver_newton(R_00, alpha, k, k_0):
    S_canonical_init = np.eye(k)
    schur_canonical_init = np.eye(k)
    A_init = np.zeros((k, k_0))

    initial_guess = wrapper(S_canonical_init, schur_canonical_init, A_init)
    solution_flat = fsolve(fixed_point_system, initial_guess, args=(R_00, alpha, k, k_0),
                           epsfcn=1e-10)
    S_cononical, schur_cononical, A = unwrap(solution_flat, k , k_0)
    S = S_cononical.T @ S_cononical
    schur = schur_cononical.T @ schur_cononical
    R = schur_decomposition(R_00, A, schur)
    return S, R








def fixed_point_solver_iterative(R_00, alpha, k, k_0, max_iter=1000, tol=1e-5):
    vars_init = wrapper(np.eye(k), np.zeros((k, k_0)), np.eye(k))

    for i in range(max_iter):
        print('iterative solver iteration', i)
        print('     vars_init are', vars_init)
        fp_init = fixed_point_system(vars_init, R_00, alpha, k, k_0)
        print('     fp_values are', fp_init)
        if np.linalg.norm(fp_init, ord=np.inf) < tol:
            break
        vars_init = fp_init + vars_init

    return vars_init

