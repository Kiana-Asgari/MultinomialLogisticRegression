"""
Solving the fixed point system.
"""

def fixed_point_solver(R_00, alpha, k, k_0):
    initial_guess = wrapper(np.eye(k), np.zeros((k, k_0)), np.eye(k))
    solution_flat = fsolve(fp_system_flatten, initial_guess, args=(R_00, alpha, k, k_0))
    S, R_10, R_11 = unwrap(solution_flat)

    return S, R_10, R_11
