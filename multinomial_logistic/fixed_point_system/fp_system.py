from multinomial_logistic.fixed_point_system.fp_integrands import batched_fixed_point_integrands
from multinomial_logistic.utils import unwrap, sqrtm, wrapper
from multinomial_logistic.integration import quadrature_integration, schur_decomposition
import numpy as np

"""
vars = [S, A, schur]
A = R_10 @ R_00^{-1/2}
schur = R_11 - R_10 @ R_00^{-1} @ R_10.T
"""

def fixed_point_system(vars, R_00, lambda_reg, alpha, k, k_0):
    #print("in fixed point system...")
    S, A, schur = unwrap(vars, k, k_0)
    fp_eqs = quadrature_integration(batched_fixed_point_integrands, S, A, schur, R_00, lambda_reg, alpha, k, k_0)

    R_00, R_10, R_11 = schur_decomposition(R_00, A, schur)

    fp_eqs1 = alpha * S @ fp_eqs[0] @ S  - schur 
    fp_eqs2 = fp_eqs[1] + 2*lambda_reg * R_10
    fp_eqs3 = fp_eqs[2] + 2*lambda_reg * R_11

    return wrapper(fp_eqs1, fp_eqs2, fp_eqs3)









"""
def fixed_point_system(vars, R_00, alpha, k, k_0):
    print("in fixed point system...")
    #S, R_10, R_11 = unwrap(vars, k, k_0)
    S_canonical, schur_canonical, A = unwrap(vars, k, k_0)
    print('  S_canonical, schur_canonical, A', S_canonical, schur_canonical, A)
  

    fp_eqs = quadrature_integration(batched_fixed_point_integrands, S_canonical, A, R_00, schur_canonical, alpha, k, k_0)

    S = S_canonical.T @ S_canonical
    schur = schur_canonical.T @ schur_canonical

    R = schur_decomposition(R_00, A, schur)
    print('         with paramsS, S, R:', S, R)

    fp_eqs1 = alpha *  fp_eqs[0]  - np.linalg.inv(S) @ schur @ np.linalg.inv(S)
    fp_eqs2 = fp_eqs[1]
    fp_eqs3 = fp_eqs[2]
    print('fp_eqs are computed:', fp_eqs1.flatten())
    print('                     ', fp_eqs2.flatten())
    print('                     ', fp_eqs3.flatten())

    return wrapper(fp_eqs1, fp_eqs2, fp_eqs3)
"""
