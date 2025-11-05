from configs.config_loader import example_configs
from multinomial_logistic.log_data.log_fp import run_and_log_fp
#from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests
from multinomial_logistic.log_data.utils import plot_errors_vs_alpha, plot_density
from multinomial_logistic.log_data.log_esd import run_and_log_esd
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


if __name__ == "__main__":
    k = 4
    k_0 = 4
    type_3 = 'two_vs_two_vs_one'
    R00 = get_R_00(k, type_3)
    run_and_log_fp(k=k, k_0=k_0, lambda_reg=0, tol=1e-4, max_iter=100, type_3=type_3,
    alphas = [4.2])

    



  




    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-5, max_iter=200,
    #         type_3=type_3, alphas = np.arange(5, 8, 0.1)[::-1],
    #         integral_mesh_size=integral_mesh_size, integral_size=integral_size)














    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=tol, max_iter=max_iter, 
    #                  type_3='three_classes_close', alphas = [5.4,5.5,5.6, 5.8, 6], 
    #                  integral_mesh_size=integral_mesh_size, integral_size=integral_size)
    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=tol, max_iter=max_iter,
    #                 type_3='symmetric', alphas = [4.3,4.4, 4.5, 4.8, 5, 5.3, 5.8, 6],
    #                 integral_mesh_size=integral_mesh_size, integral_size=integral_size)
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3='three_classes_close')
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3='symmetric')
    
    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=tol, max_iter=max_iter,
    #                 type_3='two_classes_close', alphas = [4.3,4.4, 4.5, 4.8, 5, 5.3, 5.8, 6], 
    #                 integral_mesh_size=integral_mesh_size, integral_size=integral_size)


    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3='two_classes_close')

    # plot_errors_vs_alpha(k_0=k_0, k=k, alpha_max=14, alpha_min=3.5)


