import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.special import softmax, logsumexp, expit
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature


def batched_product(S, B_batched): # S is k x k, B_batched is N x k x k. Returns N x k x k.
    return np.einsum('ij,njk->nik', S, B_batched)

def batched_inv(J_batched): # J_batched is N x k x k. Returns N x k x k.
    return np.linalg.inv(J_batched)

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


def batched_mlogit_jacobian(beta):
    v = batched_mlogit(beta)[:, :-1]
    N, k = v.shape
    # This results in an (N, k, k) array
    outer_products = v[:, :, np.newaxis] * v[:, np.newaxis, :]
    # This results in an (N, k, k) array with diagonals set to v and other elements zero
    diagonals = np.zeros_like(outer_products)
    np.einsum('ijj->ij', diagonals)[:] = v
    result_matrices = diagonals - outer_products
    return result_matrices

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


def log_sum_exp_batch(V):
    exp_V = np.exp(V)
    sum_exp_V = np.sum(exp_V, axis=1)
    sum_exp_plus_1 = 1 + sum_exp_V
    result = np.log(sum_exp_plus_1) 
    return result
