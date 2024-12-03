"""
Multivariate Integration
:param func: A k*k matrix-valued multivariable function times the desired density.
:param S, R_10, R_11, R_00, alpha, k, k_0: Constants used in the integration.
:return: A k*k matrix of integrated values.
"""

def integration(func, S, A, R_00, schur_root, alpha, k, k_0):
    expectations, err = cubature(func, args=(S, A, R_00, schur_root, alpha, k, k_0,), ndim=k+k_0 ,# vectorized=True,
                            fdim=k*k ,xmin=[-5]*(k+k_0), xmax=[5]*(k+k_0), abserr = 1e-3, relerr=1e-3, maxEval=60000, norm=2)
    return expectations.reshape((k, k))
