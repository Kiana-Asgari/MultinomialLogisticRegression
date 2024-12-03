"""
Calculating the fixed point system; returns a flattened array of 3 * k * k elements.
"""

def fp_system_flatten(vars, R_00, alpha, k, k_0):
    S, R_10, R_11 = unwrap(vars, k, k_0)
    schur = schur_complement(R_10, R_11, R_00)
    schur_root = sqrtm(schur)
    A = R_10 @ np.linalg.inv(sqrtm(R_00))

    fp_eq1 = alpha * S @ integration(func, S, A, R_00, schur_root, alpha, k, k_0) @ S - schur
    fp_eq2 = integration(func, S, A, R_00, schur_root, alpha, k, k_0)
    fp_eq3 = integration(func, S, A, R_00, schur_root, alpha, k, k_0)

    return wrapper(fp_eq1, fp_eq2, fp_eq3)
