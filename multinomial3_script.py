from configs.config_loader import example_configs
from multinomial_logistic.log_data.log_fp import run_and_log_fp, refine_logged_fp
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests_regularized, refine_logged_fp_tests_regularized
from multinomial_logistic.log_data.utils import plot_errors_vs_alpha, plot_density, plot_regularized_error
from multinomial_logistic.log_data.log_fp_regularized import run_and_log_fp_regularized, refine_logged_regulairzed_fp
from multinomial_logistic.log_data.log_regularized_mle_empirical import run_and_log_mle_regularized
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests, refine_logged_fp_tests
from multinomial_logistic.log_data.log_regularized_mle_empirical import run_and_log_mle_regularized
import numpy as np
import torch
from configs.R_initiation import get_R_00
import sys
from multinomial_logistic.ESD.Marchenko_Pastur_FP_GPU import stieltjes_inversion
from scipy.linalg import sqrtm
from multinomial_logistic.log_data.log_esd import run_and_log_esd

def _ESD():

    type_3 = 'two_vs_two_vs_one'
    k,k_0 = 4, 4
    R00 = get_R_00(k, type_3)
    z_imag = 1e-5

    for alpha in [4]:
        # z_real_values = np.linspace(0.0005, 0.002, 20)
        # run_and_log_esd(k_0=k_0, k=k, lambda_reg=0, alpha_input=alpha, R_00_input=R00,
        #                     z_imag=z_imag, z_real_values=z_real_values, 
        #                     tol=1e-2, max_iter=500, 
        #                     type_3=type_3)

        try:
            # from multinomial_logistic.log_data.log_mle_empirical import run_and_log_esd_empirical
            # run_and_log_esd_empirical(alpha=alpha, k=k, k_0=k_0,  R_00=R00,
            #                                        n_trials=100, d=250)
            plot_density(k=k, k_0=k_0, R_00=R00, alpha_target=alpha,type_3=type_3, emp_bins=150)


        except Exception as e:
            print(f"Error for alpha {alpha}: {e}")





def ref_reg(alpha=0, reg_min=None, reg_max=None):
    """ the final plot"""
    type_3 = 'symmetric'
    k,k_0 = 4, 4
    R00 = get_R_00(k, type_3)
    plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=250, n_trials=100, lambda_reg_max=0.31, lambda_reg_min=0)

def ref_alpha():
    k,k_0 = 4, 4
    d=250
    n_trials=100
    plot_errors_vs_alpha(k=k, k_0=k_0, alpha_max=12, alpha_min=3.6, d=d, 
        n_trials=n_trials, desired_metric='all')


if __name__ == "__main__":
    _ESD()
    # ref_alpha()
    # ref_reg()















