from multinomial_logistic.fixed_point_system.fp_integrands import batched_fixed_point_integrands
from multinomial_logistic.utils import unwrap, schur_complement, sqrtm, wrapper
from multinomial_logistic.integration import quadrature_integration
import numpy as np

def fixed_point_system(vars, R_00, alpha, k, k_0):
    print("in fixed point system...")
    S, R_10, R_11 = unwrap(vars, k, k_0)
    schur = schur_complement(R_10, R_11, R_00)
    schur_root = sqrtm(schur)
    A = R_10 @ np.linalg.inv(sqrtm(R_00))

    fp_eqs = quadrature_integration(batched_fixed_point_integrands, S, A, R_00, schur_root, alpha, k, k_0)
    fp_eqs1 = alpha * S @ fp_eqs[0] @ S - schur
    fp_eqs2 = fp_eqs[1]
    fp_eqs3 = fp_eqs[2]
    print('fp_eqs are computed with shapes', fp_eqs1.shape, fp_eqs2.shape, fp_eqs3.shape)
    return wrapper(fp_eqs1, fp_eqs2, fp_eqs3)
