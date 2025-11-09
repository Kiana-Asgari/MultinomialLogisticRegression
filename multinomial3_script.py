from configs.config_loader import example_configs
from multinomial_logistic.log_data.log_fp import run_and_log_fp, refine_logged_fp
#from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests_regularized, refine_logged_fp_tests_regularized
from multinomial_logistic.log_data.utils import plot_errors_vs_alpha, plot_density, plot_regularized_error
from multinomial_logistic.log_data.log_esd import run_and_log_esd
from multinomial_logistic.log_data.log_fp_regularized import run_and_log_fp_regularized, refine_logged_regulairzed_fp
from multinomial_logistic.log_data.log_regularized_mle_empirical import run_and_log_mle_regularized

import numpy as np
import torch
from configs.R_initiation import get_R_00
import sys
from multinomial_logistic.ESD.Marchenko_Pastur_FP_GPU import stieltjes_inversion
from scipy.linalg import sqrtm



def _ESD():
    for alpha in [15]:
        # eigenvalues = esd_empirical(alpha=alpha, k=k, lambda_reg=0, R_00=R00,
        #                             n_trials=2, d=300, skitlearn=True)
        # plot_density(k=k, k_0=k_0, R_00=R00, alpha_target=alpha, eigenvalues=eigenvalues,
        #                             emp_histogram=True, emp_bins=120)
        try:
            run_and_log_esd(k_0=k_0, k=k, lambda_reg=0, alpha_input=alpha, R_00_input=R00,
                                z_imag=1e-5, z_real_values=z_real_values, max_iter=6200, 
                                type_3=type_3)
        except Exception as e:
            print(f"Error for alpha {alpha}: {e}")

def reg_stuff():
    k, k_0 = 4, 4
    type_3 = 'symmetric'
    R00 = get_R_00(k, type_3)
    alphas = [1.5, 3, 5, 10]
    lambda_regs = 1/2 * np.concatenate([np.arange(0.06, 0.12, 0.005)])
    run_and_log_fp_regularized(k_0=k_0, k=k, lambda_regs=lambda_regs, alphas=alphas, tol=1e-5, max_iter=80, type_3=type_3, integral_mesh_size=8, integral_size=7)
    run_and_log_fp_tests_regularized(k_0=k_0, k=k, R_00=R00, type_3=type_3)


    for modified_alpha in [1.5, 3, 5, 10]:# 5, 10]:     
        #refine_logged_regulairzed_fp(k_0=k_0, k=k, type_3=type_3, tol=1e-5, max_iter=10, integral_size=7, integral_mesh_size=10,    modified_alpha=modified_alpha)
        refine_logged_fp_tests_regularized(k_0=k_0, k=k, R_00=R00, type_3=type_3, metric_name='misclassification_test_errors', modified_alpha=modified_alpha)


    plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=300, n_trials=100, lambda_reg_max=0.15, lambda_reg_min=0)



if __name__ == "__main__":
    k = 4
    k_0 = 4
    type_3 = 'two_vs_two_vs_one'
    reg_stuff()

    R00 = get_R_00(k, type_3)

    # refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=1, max_iter=2, type_3=type_3, integral_mesh_size=13)
    # refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=1, max_iter=2, type_3=type_3, integral_mesh_size=13)

    # type_3 = 'two_classes_close'
    # R00 = get_R_00(k, type_3)

    # refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=1, max_iter=2, type_3=type_3)
    # refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=1, max_iter=2, type_3=type_3)


    # run_and_log_fp_tests(k=k, k_0=k_0, lambda_reg=0, type_3=type_3)
    # plot_errors_vs_alpha(k=k, k_0=k_0,alpha_min=3.0, alpha_max=14.0)

    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-5, max_iter=200,
    #         type_3=type_3, alphas = np.arange(5, 8, 0.1)[::-1],
    #         integral_mesh_size=integral_mesh_size, integral_size=integral_size)






