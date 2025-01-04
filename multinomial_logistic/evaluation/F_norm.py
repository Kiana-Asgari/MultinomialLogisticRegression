import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

from multinomial_logistic.utils import mlogit, batched_mlogit, log_sum_exp_batch
from cubature import cubature
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_scalar_mult, batched_normal_basis
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.utils import batched_sqrtm 
from multinomial_logistic.evaluation.utils import plot_array
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline





def plot_norm_vs_lambda_reg(R_00, lambda_reg_min, lambda_reg_max ,k, k_0, max_iter, save_path):

    print('plotting norm vs alpha... for parameters:')
    print('     R_00 = ', R_00)
    alpha_values = [2, 4]               
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, max_iter, endpoint=False)

    norm_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    empirical_norm_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))

    legends = ['alpha=2', 'alpha=4']

    for i, alpha in enumerate(alpha_values):
        for j, lambda_reg in enumerate(lambda_reg_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                            lambda_reg=lambda_reg/2, alpha=alpha, k=k, k_0=k_0)
            theta_hat, empirical_norm, _ = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=lambda_reg,\
                                                    d=250, R_00=R_00, n_trials=100)
            R_11 = schur + R_01 @ np.linalg.inv(R_00) @ R_01.T

            if divergence:
                print('     **Divergence detected**')
                break

            print(f'     R_11 = {R_11}')
            print(f'     R_01 = {R_01}')
            print(f'     R_00 = {R_00}')

            norm_batches[i,j] = np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T)
            empirical_norm_batches[i,j] = empirical_norm

            print(f' alpha = {alpha:.2f}, lambda_reg = {lambda_reg:.2f}   theoretical norm = {norm_batches[i,j]}')

            print(f' alpha = {alpha:.2f}, lambda_reg = {lambda_reg:.2f}   empirical norm = {empirical_norm_batches[i,j]}')

    title = f"||theta-theta_0||_F vs lambda_reg, number of class={k+1:d},R_00= ({R_00})"
    name = f"F_norm_vs_lambda_reg_nclass={k+1:d}_R_00={R_00}" 
    plot_array( lambda_reg_values, norm_batches, \
                empricial_data_batch=empirical_norm_batches, legends=legends, \
                title=title, \
                x_label="lambda_reg", y_label="F_norm", \
                name=name, save_path=save_path)
    