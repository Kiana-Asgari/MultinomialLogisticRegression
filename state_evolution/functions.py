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