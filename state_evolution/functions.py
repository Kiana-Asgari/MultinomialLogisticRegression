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