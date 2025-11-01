import numpy as np
from cubature import cubature
from multinomial_logistic.utils import unwrap, batched_mult, batched_outer
from scipy.linalg import sqrtm

def coloring_transform(Z_vectorized, A, R_00, schur_root, alpha, k, k_0):
    # A = R_10 @ R_00^{-1/2}, schur_root = sqrtm(R_11 - R_10 @ R_00^{-1} @ R_10.T)
    Z_top = Z_vectorized[:,:k] # (N, k)
    Z_bottom = Z_vectorized[:,-k_0:] # (N, k_0)
    g = batched_mult(A , Z_bottom) + batched_mult(schur_root, Z_top)
    g_0 = batched_mult(sqrtm(R_00), Z_bottom)

    return g, g_0

def schur_complement(R_10, R_11, R_00): # Returns R\R_00
    R_00inv = np.linalg.inv(R_00)
    return R_11 - R_10 @ R_00inv @ (R_10).T

def schur_decomposition(R_00, A, schur): # Returns R_10, R_11, R_00
    R_11 = schur + A @ A.T
    R_10 = A @ sqrtm(R_00)
    return R_00, R_10, R_11
    
def quadrature_integration(func,S, A, schur, R_00, lambda_reg, alpha, k, k_0):
    #print('     integrating... ')
    expectations, err = cubature(func, args=(S, A, schur, R_00, lambda_reg, alpha, k, k_0), ndim=k+k_0,
                                  vectorized=True,
                                  fdim=k*k + k*k + k*k_0  ,xmin=[-20]*(k+k_0), xmax=[20]*(k+k_0), abserr = 1e-4, relerr=1e-4,
                                  maxEval= k * 1_500_000, norm=2)
    #print('     expectations are calculated with shape', expectations.shape)
    return unwrap(expectations, k, k_0)




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