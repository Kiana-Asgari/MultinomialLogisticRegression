import numpy as np
from cubature import cubature

def quadrature_integration(func, S, A, R_00, schur_root, alpha, k, k_0):
    print('integrating ')
    expectations, err = cubature(func, args=(S, A, R_00, schur_root, alpha, k, k_0,), ndim=k+k_0,
                                  vectorized=True,
                                  fdim=k*k ,xmin=[-8]*(k+k_0), xmax=[8]*(k+k_0), abserr = 1e-3, relerr=1e-3,
                                  maxEval=100000, norm=2)
    return expectations.reshape((k, k))




"""
def monte_carlo_integration(func, S, A, R_00, schur_root, alpha, k, k_0):
    Z = np.random.multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0), size=5000)

    results = []
    for z in Z:
        val = func(z, S, A, R_00, schur_root, alpha, k, k_0)
        results.append(np.array(val).flatten())

    results = np.array(results)
    expectation = np.mean(results, axis=0)

    # Reshape back to (3, k, k)
    return expectation.reshape((3, k, k))
"""