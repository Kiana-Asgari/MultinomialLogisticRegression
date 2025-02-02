import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import mlogit, batched_mlogit, log_sum_exp_batch
from cubature import cubature
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_scalar_mult, batched_normal_basis
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.evaluation.utils import plot_array
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline


def plot_train_log_loss_vs_lambda(R_00, lambda_reg_min, lambda_reg_max, k, k_0, max_iter, save_path):
    print('plotting train error vs lambda_reg... for parameters:')

    alpha_values = [2,4,6]               
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, max_iter, endpoint=False)

    train_error_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    train_error_empirical_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    legends = ['alpha=2', 'alpha=4', 'alpha=6']


    for i, alpha in enumerate(alpha_values):
        for j, lambda_reg in enumerate(lambda_reg_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=lambda_reg/2, alpha=alpha, k=k, k_0=k_0)
            if divergence:
                print('     **Divergence detected**')
                break
            train_error_batches[i,j] = train_error(R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=alpha, k=k, k_0=k_0)
            _, _, _, train_error_empirical = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=lambda_reg,\
                                                    d=250, R_00=R_00, n_trials=100)
            train_error_empirical_batches[i,j] = train_error_empirical   
            print(f' alpha = {alpha:.2f}   train_error_batches = {train_error_batches[i,j]}', 'empirical = ', train_error_empirical_batches[i,j])

    title = f"train error log loss vs lambda_reg, number of class={k+1:d}, R_00=({R_00})"
    name = f"train_error_vs_lambda_reg_nclass={k+1:d}_R_00={R_00}" 
    plot_array(lambda_reg_values, train_error_batches,   train_error_empirical_batches,\
               legends=legends, title=title,\
                x_label="lambda_reg", y_label="train error",\
                name=name, save_path=save_path)







def plot_train_log_loss_vs_alpha(R_00, alpha_min, alpha_max, k, k_0, max_iter, save_path):
    print('plotting train error vs alpha... for parameters:')

    R_00_values = [np.array([[1,0],[0,1]])]               
    alpha_values = np.linspace(alpha_min, alpha_max, max_iter, endpoint=False)

    train_error_batches = np.zeros((len(R_00_values), len(alpha_values)))
    train_error_empirical_batches = np.zeros((len(R_00_values), len(alpha_values)))
    #train_error_batches[1,:] =  np.array([0.66755431, 0.74089194, 0.7844488, 0.81359218, 0.83453579, 0.85033963,\
    # 0.86269969, 0.87263667, 0.88080196, 0.88763244, 0.89343165, 0.89841735,\
    # 0.90275024, 0.9065508, 0.90991187])
    #train_error_empirical_batches[1,:] =  np.array([0.66812576, 0.74323643, 0.78666792, 0.81369186, 0.83489762, 0.85040574,\
    # 0.90239719, 0.90498782, 0.90966888])
    legends = [f'R_00=I']
    
    for i, R_00 in enumerate(R_00_values):
        for j, alpha in enumerate(alpha_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=0, alpha=alpha, k=k, k_0=k_0)
            if divergence:
                print('     **Divergence detected**')
                break
            train_error_batches[i,j] = train_error(R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=alpha, k=k, k_0=k_0)
            _, _, _, train_error_empirical = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=0,\
                                                    d=250, R_00=R_00, n_trials=100)
            train_error_empirical_batches[i,j] = train_error_empirical   
            print(f' alpha = {alpha:.2f}   train_error_batches = {train_error_batches[i,j]}', 'empirical = ', train_error_empirical_batches[i,j])
    print('train_error_batches: ', train_error_batches)
    print('train_error_empirical_batches: ', train_error_empirical_batches)
    title = f"train error log loss vs alpha, number of class={k+1:d}, R_00=({R_00})"
    name = f"train_error_vs_alpha_nclass={k+1:d}_R_00={R_00}" 
    plot_array(alpha_values, train_error_batches, train_error_empirical_batches,\
                legends=legends, title=title,\
               x_label="alpha", y_label="log loss",\
                name=name, save_path=save_path)
    





def train_error(R_00, schur, R_01, S,alpha, k, k_0, seed=42):
    np.random.seed(seed)
    print('     computing train  error... with parameters: R_00=', R_00, 'schur=', schur, 'R_01=', R_01, 'S=', S)
    loss = integrate(_train_log_loss_integrand, R_00, schur, R_01, S, alpha, k, k_0)
    print('     done computing train error with loss: ', loss)
    return loss



#########################
# Log loss integrand
#########################
def _train_log_loss_integrand(Z_batch, R_00, schur, R_01, S, alpha, k, k_0):
    # Compute schur complement
    N = Z_batch.shape[0]
    R_00_inv = np.linalg.inv(R_00)
    schur_root = sqrtm(schur)
    A = R_01 @ sqrtm(R_00_inv)
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    prob_y_batch = batched_mlogit(g_0_batch)
    
    loss = np.zeros(N)
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        prox_g_batch, div_prox = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S) # prox(g + yS; S)

        
        if div_prox:
            print('     **prox Divergence detected**')
            break
        logloss_batch = log_sum_exp_batch(prox_g_batch) - np.einsum('ij,ij->i', y_batch, prox_g_batch)
        loss += logloss_batch * prob_y_batch[:, i] # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    return loss * pdf
    
   
    



def integrate(integrand, R_00, schur, R_01, S, alpha, k, k_0):
    #print('     integrating... ')
    fdim = 1
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(R_00, schur, R_01, S, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-3.6]*ndim, xmax=[3.6]*ndim, abserr = 1e-5,
                                  maxEval=1_500_000, norm=2)
    if err.item() > 1e-4:
        print('     **[Warning] train error integration error is too large**, err=', err)
    #print('     done integrating')
    return expectations     
