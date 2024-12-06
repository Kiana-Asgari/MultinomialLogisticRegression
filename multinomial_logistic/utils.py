import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.special import softmax, logsumexp, expit
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature


def batched_scalar_mult(S_batched, p_batch): # S_batched is N x k x k, p_batch is N 1D array. Returns N x k
    return S_batched * p_batch[:, np.newaxis, np.newaxis]

def batched_mult(S, y_batch): # S is k x k, y_batch is N x k
    return np.dot(S, y_batch.T).T

def batched_outer(x_batch, y_batch): # x_batch is N x k, y_batch is N x k
    return np.einsum('ni,nj->nij', x_batch, y_batch)

def unwrap(vars, k, k_0): # unwrap (S, R_10, R_11).flatten()

    vars = np.array(vars)
    assert len(vars) == k*k + k*k_0 + k*k
    S = vars[:k*k].reshape((k, k))
    R_10 = vars[k*k:k*k + k*k_0].reshape((k, k_0))
    R_11 = vars[k*k + k*k_0:].reshape((k, k))
    return S, R_10, R_11

def wrapper(S, R_10, R_11): # wrap (S, R_10, R_11)
    return np.concatenate((S.flatten(), R_10.flatten(), R_11.flatten()))

def mlogit(beta): # returns a k+1 dimentional logistic perobablity vector with the prob of 0 as the last element
    return softmax(np.append(beta,0))

def schur_complement(R_10, R_11, R_00): # Returns R\R_00
    R_00inv = np.linalg.inv(R_00)
    return R_11 - R_10 @ R_00inv @ (R_10).T

def batched_normal_basis(i, k, N): # Reutrns the standard basis of R^k
    assert i in range(-1, k)
    y = np.zeros(k) if i == -1 else np.eye(k)[:, i]
    return np.tile(y, (N, 1))


def _softmax(x):
    # Softmax function for each row in a 2D array (batch processing)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def batched_mlogit(beta):
    # Compute the logistic probabilities, including the last element as 0
    return _softmax(np.hstack([beta, np.zeros((beta.shape[0], 1))]))  # Append 0 for the last element


def batched_wrapper(integrand1, integrand2, integrand3, k, k_0):
    """
    Wrap three N*k*k integrands into a N*(3k²) matrix
    """
    N = integrand1.shape[0]
    return np.concatenate([
        integrand1.reshape(N, -1),
        integrand2.reshape(N, -1),
        integrand3.reshape(N, -1)
    ], axis=1)
