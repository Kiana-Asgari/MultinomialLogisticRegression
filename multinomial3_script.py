from configs.config_loader import example_configs
from multinomial_logistic.log_data.log_fp import run_and_log_fp, refine_logged_fp
#from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests_regularized, refine_logged_fp_tests_regularized
from multinomial_logistic.log_data.utils import plot_errors_vs_alpha, plot_density, plot_regularized_error
from multinomial_logistic.log_data.log_esd import run_and_log_esd
from multinomial_logistic.log_data.log_fp_regularized import run_and_log_fp_regularized, refine_logged_regulairzed_fp
from multinomial_logistic.log_data.log_regularized_mle_empirical import run_and_log_mle_regularized
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests, refine_logged_fp_tests

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
            print(f"Error for alpha {alpha}: {e}", flush=True)

def reg_stuff():
    k, k_0 = 4, 4
    type_3 = 'symmetric'
    R00 = get_R_00(k, type_3)
    for modified_alpha in [1.5, 3, 5, 10]:     
        refine_logged_regulairzed_fp(k_0=k_0, k=k, type_3=type_3, tol=1e-5, max_iter=2, integral_size=5, integral_mesh_size=13, 
                                    modified_alpha=modified_alpha, dtype=torch.float32)
        refine_logged_fp_tests_regularized(k_0=k_0, k=k, R_00=R00, type_3=type_3, metric_name='misclassification_test_errors', modified_alpha=modified_alpha)
        plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=300, n_trials=100, lambda_reg_max=0.06, lambda_reg_min=0)

    plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=300, n_trials=100, lambda_reg_max=0.06, lambda_reg_min=0)



if __name__ == "__main__":
    k = 4
    k_0 = 4
    type_3 = 'symmetric'
    R00 = get_R_00(k, type_3)
    reg_stuff()

    #plot_errors_vs_alpha(k=k, k_0=k_0, alpha_min=6, alpha_max=13.1)
    # type_3 = 'three_classes_close'
    # alphas = [7.5, 6.5, 5.5, 4.5, 4.4]
    # run_and_log_fp(k_0=k_0, k=k, alphas=alphas, lambda_reg=0, tol=1e-2, max_iter=50, use_lambda_reg=0.000, dtype=torch.float32,
    #                     type_3=type_3, integral_mesh_size=13, integral_size=5)
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='misclassification_test_error', alpha_min=3.6, alpha_max=10)
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='misclassification_test_error', alpha_min=3.6, alpha_max=10)





    for i in range(10):
        print(f"\n\nIteration {i}\n\n", flush=True)
        type_3 = 'three_classes_close'
        R00 = get_R_00(k, type_3)
        refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=5*1e-2, max_iter=5, use_lambda_reg=0.000, dtype=torch.float32,
                        type_3=type_3, integral_mesh_size=13, integral_size=5, alpha_max=14, alpha_min=3.6)
        refine_logged_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='misclassification_test_error', alpha_min=3.6, alpha_max=10)
        #refine_logged_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='test_error', alpha_min=3.6, alpha_max=10)
        plot_errors_vs_alpha(k=k, k_0=k_0, alpha_min=0, alpha_max=14.0)
        continue
 

        type_3 = 'two_classes_close'
        R00 = get_R_00(k, type_3)
        refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=5*1e-2, max_iter=5, use_lambda_reg=0.000, dtype=torch.float32,
                        type_3=type_3, integral_mesh_size=13, integral_size=5, alpha_max=5, alpha_min=3.6)
        refine_logged_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='misclassification_test_error', alpha_min=3.6, alpha_max=10)
        type_3 = 'symmetric'
        R00 = get_R_00(k, type_3)
        refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=5*1e-2, max_iter=5, use_lambda_reg=0.000, dtype=torch.float32,
                        type_3=type_3, integral_mesh_size=13, integral_size=5, alpha_max=5, alpha_min=3.6)
        refine_logged_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='misclassification_test_error', alpha_min=3.6, alpha_max=10)
    
        type_3 = 'two_vs_two_vs_one'
        R00 = get_R_00(k, type_3)
        refine_logged_fp(k_0=k_0, k=k, lambda_reg=0, tol=5*1e-2, max_iter=5, use_lambda_reg=0.000, dtype=torch.float32,
                        type_3=type_3, integral_mesh_size=13, integral_size=5, alpha_max=5, alpha_min=3.6)
        refine_logged_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3=type_3, metric_name='misclassification_test_error', alpha_min=3.6, alpha_max=10)

        plot_errors_vs_alpha(k=k, k_0=k_0, alpha_min=3.6, alpha_max=14.0)

            



    plot_errors_vs_alpha(k=k, k_0=k_0, alpha_min=3.6, alpha_max=14.0)







