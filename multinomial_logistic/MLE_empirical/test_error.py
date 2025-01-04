import numpy as np
from cubature import cubature
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import batched_mlogit, batched_normal_basis, log_sum_exp_batch
from multinomial_logistic.integration import coloring_transform



def test_error(Theta_0, Theta_hat):
    R_00 = Theta_0 @ Theta_0.T
    R_11 = Theta_hat @ Theta_hat.T
    R_01 = Theta_0 @ Theta_hat.T
    schur = R_11 - R_01 @ np.linalg.inv(R_00) @ R_01.T
    A = np.linalg.inv(R_00) @ R_01.T
    k_0 = Theta_0.shape[0]
    k = Theta_hat.shape[0]
    alpha = None
    return integration(_test_error_integrand, R_00, schur, A, alpha, k, k_0)



def _test_error_integrand(Z_batch, R_00, schur, A, alpha, k, k_0):
    #print('     computing test error integrand...')
    #print('     Z_batch shape:', Z_batch.shape)
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur)
    g_1_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)

    # Compute the loss
    prob_y_batch = batched_mlogit(g_0_batch)

    loss = np.zeros(N)
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        logloss_batch = log_sum_exp_batch(g_1_batch) - np.einsum('ij,ij->i', y_batch, g_1_batch)

        loss += logloss_batch * prob_y_batch[:, i] # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    return loss * pdf








####
def integration(integrand, R_00, schur, A, alpha, k, k_0):
    fdim = 1
    ndim = k+k_0
    expectations, err = cubature(integrand, args=( R_00, schur, A, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-3.6]*ndim, xmax=[3.6]*ndim, abserr=1e-4,
                                  maxEval= 500_000, norm=2)
    #for e in err:
    #   if e > 1e-3:
    #     print('     **[Warning] state evolution integration error is too large**')
    #     break
    #print('     done integrating')
    return expectations