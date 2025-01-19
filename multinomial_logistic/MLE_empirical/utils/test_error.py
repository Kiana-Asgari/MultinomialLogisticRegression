import numpy as np
from cubature import cubature
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import batched_mlogit, batched_normal_basis, log_sum_exp_batch
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.evaluation.misclassification_test_error import misclassification_test_error
from sklearn.metrics import accuracy_score
from multinomial_logistic.MLE_empirical.utils.data_generation import generate_data


def mle_misclassification_test_error(Theta_0, Theta_hat):
    k = Theta_0.shape[0]
    d = Theta_0.shape[1]
    X_test, y_test_onehot = generate_data(alpha=1e3, d=d, k=k, Theta_0=Theta_0)
    y_test = np.argmax(y_test_onehot, axis=1) + np.max(y_test_onehot, axis=1)
    theta_tilde = np.vstack([np.zeros((1, Theta_hat.shape[1])), Theta_hat])
    misclass_test_error_skit = 1-accuracy_score(y_test, np.argmax(X_test @ theta_tilde.T, axis=1))

    R_00 = Theta_0 @ Theta_0.T
    R_11 = Theta_hat @ Theta_hat.T
    R_01 = Theta_0 @ Theta_hat.T
    schur = R_11 - R_01.T @ np.linalg.inv(R_00) @ R_01.T
    A = R_01.T @ sqrtm(np.linalg.inv(R_00))
    k_0 = Theta_0.shape[0]
    k = Theta_hat.shape[0]
    misclass_test_error = 0 #misclassification_test_error(S=None, R_00=R_00, schur_t=schur,    
                                         #A_t=A, alpha=None, k=k, k_0=k_0, monte_carlo=False)
    print('     misclass_test_error: ', misclass_test_error, '  ,[skit]: ', misclass_test_error_skit)

    return misclass_test_error_skit

    return misclass_test_error

def test_error(Theta_0, Theta_hat):
    R_00 = Theta_0 @ Theta_0.T
    R_11 = Theta_hat @ Theta_hat.T
    R_01 = Theta_0 @ Theta_hat.T
    schur = R_11 - R_01 @ np.linalg.inv(R_00) @ R_01.T
    A = R_01.T @ sqrtm(np.linalg.inv(R_00))
    k_0 = Theta_0.shape[0]
    k = Theta_hat.shape[0]
    alpha = None
    test_error = integration(_test_error_integrand, R_00, schur, A, alpha, k, k_0)
    print('     test_error: ', test_error)
    return test_error



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


def _misclassification_test_error_integrand(Z_batch, R_00, schur, A, alpha, k, k_0):
    #print('     computing test error integrand...')
    #print('     Z_batch shape:', Z_batch.shape)
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur)
    g_1_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)

    # Compute the loss
    prob_y_batch = batched_mlogit(g_0_batch)

    loss = np.ones(N)
    for i in range(-1, k):
        loss += prob_y_batch[:, i]@prob_y_batch[:, i]

    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    return loss * pdf






####
def integration(integrand, R_00, schur, A, alpha, k, k_0):
    fdim = 1
    ndim = k+k_0
    expectations, err = cubature(integrand, args=( R_00, schur, A, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-4]*ndim, xmax=[4]*ndim, abserr=1e-5, relerr=1e-5,
                                  maxEval= 15_000_000, norm=2)
    #for e in err:
    if err.item() > 1e-5:
        print('     **[Warning] state evolution integration error is too large**,', err)
    #print('     done integrating')
    return expectations