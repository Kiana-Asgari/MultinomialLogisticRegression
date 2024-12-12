"""
Solving the fixed point system.
"""
import numpy as np
from scipy.optimize import fsolve
from multinomial_logistic.utils import wrapper, unwrap
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system

def fixed_point_solver_newton(R_00, alpha, k, k_0):
    initial_guess = wrapper(10*np.eye(k), np.zeros((k, k_0)), np.eye(k))
    solution_flat = fsolve(fixed_point_system, initial_guess, args=(R_00, alpha, k, k_0))
    S, R_10, R_11 = unwrap(solution_flat, k , k_0)

    return S, R_10, R_11

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

