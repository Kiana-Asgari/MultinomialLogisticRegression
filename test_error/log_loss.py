import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import mlogit, batched_mlogit, log_sum_exp_batch
from cubature import cubature
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_scalar_mult, batched_normal_basis
from state_evolution.full_recursion import state_evolution_full_recursion
from test_error.utils import plot_array


def plot_log_loss_vs_alpha(R_00, alpha_min, alpha_max, lambda_reg, k, k_0):
    alpha_values = np.linspace(alpha_min, alpha_max, 100, endpoint=True)
    log_loss_values = []
    for alpha in alpha_values:
        schur, R_01, S = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
        log_loss_values.append(test_error(R_00, schur, R_01, alpha, k, k_0))
    title = f"log loss vs alpha for lambda_reg={lambda_reg:.2f}, number of class={k+1:d}"
    name = f"log_loss_vs_alpha_lambda_reg={lambda_reg:.2f}_nclass={k+1:d}" 
    plot_array(alpha_values, log_loss_values, title=title,\
               x_label="alpha", y_label="log loss",\
                name=name, save_path='multinomial_logistic/data/log_loss')

def plot_log_loss_vs_lambda_reg(R_00, lambda_reg_min, lambda_reg_max, alpha, k, k_0):
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, 100, endpoint=True)
    log_loss_values = []
    for lambda_reg in lambda_reg_values:
        schur, R_01, S = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
        log_loss_values.append(test_error(R_00, schur, R_01, alpha, k, k_0))
    nclass = k+1
    title = f"log loss vs lambda_reg for alpha={alpha:.2f}, number of class={nclass:d}"
    name = f"log_loss_vs_lambda_reg_alpha={alpha:.2f}_nclass={nclass:d}" 
    print('lambda_reg_values', lambda_reg_values)
    print('log_loss_values', log_loss_values)
    plot_array(lambda_reg_values, log_loss_values, title=title,\
               x_label="lambda_reg", y_label="log loss",\
                name=name, save_path='multinomial_logistic/data/log_loss')
    return lambda_reg_values, log_loss_values



def test_error(R_00, schur, R_01, alpha, k, k_0):
    print('     computing test error... ')
    loss = integrate(log_loss, R_00, schur, R_01, alpha, k, k_0)
    print('     done computing test error')
    return loss


def log_loss(Z_batch, R_00, schur, R_01, alpha, k, k_0):
    # Compute schur complement
    N = Z_batch.shape[0]
    R_00_inv = np.linalg.inv(R_00)
    schur_root = sqrtm(schur)
    A = R_01 @ sqrtm(R_00_inv)
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)
    
    loss = np.zeros(N)
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        logloss_batch = log_sum_exp_batch(g_batch) - np.einsum('ij,ij->i', y_batch, g_batch)
        loss += logloss_batch * prob_y_batch[:, i] # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    return loss * pdf
    
   
    



def integrate(integrand, R_00, schur, R_01, alpha, k, k_0):
    #print('     integrating... ')
    fdim = 1
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(R_00, schur, R_01, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-8]*ndim, xmax=[8]*ndim, abserr = 1e-6, relerr=1e-6,
                                  maxEval=150000, norm=2)
    #if err.any() > 1e-2:
    #    print('     **[Warning] integration error is too large**')
    #print('     done integrating')
    return expectations     