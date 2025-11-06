import numpy as np
from multinomial_logistic.utils import batched_mlogit_jacobian, batched_mlogit, batched_product, batched_inv

def score_batched(V_batch, y_batch):
    """
    V_batch: N*k
    y_batch: N*k
    returns: N*k p(v) - y
    """
    return batched_mlogit(V_batch)[:,:-1] - y_batch

def score_jacobian_batched(V_batch, S, k):
    """
    V_batch: N*k
    S: k*k
    returns: N*k*k (I_k + S @ Jp(V)) ^ {-1}
    """
    J = batched_mlogit_jacobian(V_batch)
    return batched_inv(np.eye(k) + batched_product(S, J))
    # TODO: why is the order of S and J different from the paper?
    # J = batched_mlogit_jacobian(V_batch)
    # inv_score_jacobian_batch = np.einsum('nij,jl->nil', J, S) + np.eye(k)[None, :, :] 
    # return batched_inv(inv_score_jacobian_batch)

def score_jacobian_batched_with_volume_factor(V_batch, S, k):
    J = batched_mlogit_jacobian(V_batch)
    score_inv_batch = np.eye(k) + batched_product(S, J)
    return batched_inv(score_inv_batch), np.linalg.det(score_inv_batch)

def MP_batched(V_batch, MP_S_inv, k):
    """
    V_batch: N*k
    k: int
    MP_S: k*k
    returns: N*k*k (I_k + S @ Jp(V)) ^ {-1} @ Jp(V)
    """
    J = batched_mlogit_jacobian(V_batch) # N*k*k
    N = V_batch.shape[0]
    score_batch = np.einsum('ij,njl->nil', MP_S_inv,\
                             batched_inv(MP_S_inv + J )) # N*k*k
    return np.einsum('nij,njl->nil', score_batch, J, optimize='optimal') # N*k*k

def ODE_batched(V_batch, S_MP, dS_MP, k):
    J = batched_mlogit_jacobian(V_batch) # N*k*k
    N = V_batch.shape[0]
    identity_batch = np.broadcast_to(np.eye(k), (N, k, k))
    
    temp = np.einsum('nij,njl->nil', J,\
                     batched_inv(np.einsum('nij,njl->nil', S_MP, J) + identity_batch))
    return np.einsum('nij,jk ,nkl->nil', temp, dS_MP, temp, optimize='optimal')




import torch

def _matrix_sqrt(matrix):
    symmetric_matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    original_dtype = symmetric_matrix.dtype

    work_matrix = symmetric_matrix
    if original_dtype == torch.float16:
        work_matrix = symmetric_matrix.to(dtype=torch.float32)


    eigenvalues, eigenvectors = torch.linalg.eigh(work_matrix)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
    sqrt_matrix = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)

    if sqrt_matrix.dtype != original_dtype:
        sqrt_matrix = sqrt_matrix.to(dtype=original_dtype)

    return sqrt_matrix


def _matrix_inv(matrix):
    if matrix.dtype == torch.float16:
        matrix_work = matrix.to(dtype=torch.float32)
    else:
        matrix_work = matrix
    inv_matrix = torch.linalg.inv(matrix_work)
    if inv_matrix.dtype != matrix.dtype:
        inv_matrix = inv_matrix.to(dtype=matrix.dtype)
    return inv_matrix


def _solve(A, B):
    if A.dtype == torch.float16:
        A_new = A.to(dtype=torch.float32)
        B_new = B.to(dtype=torch.float32)
    else:
        A_new = A
        B_new = B
    solution = torch.linalg.solve(A_new, B_new)
    if solution.dtype != A.dtype:
        solution = solution.to(dtype=A.dtype)
    return solution