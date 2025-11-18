from configs.config_loader import example_configs
from multinomial_logistic.log_data.log_fp import run_and_log_fp, refine_logged_fp
#from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
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


def _ESD():
    from multinomial_logistic.MLE_empirical.ESD_empirical import esd_empirical
    type_3 = 'two_vs_two_vs_one'
    k,k_0 = 4, 4
    R00 = get_R_00(k, type_3)
    for alpha in [10,9,8,7,6]:
        try:
            eigenvalues = esd_empirical(alpha=alpha, k=k, lambda_reg=0, R_00=R00,
                                        n_trials=5, d=300, skitlearn=True)
            plot_density(k=k, k_0=k_0, R_00=R00, alpha_target=alpha, eigenvalues=eigenvalues,
                                        emp_histogram=True, emp_bins=120)

            run_and_log_esd(k_0=k_0, k=k, lambda_reg=0, alpha_input=alpha, R_00_input=R00,
                                z_imag=1e-5, z_real_values=[0.01], max_iter=500, 
                                type_3=type_3)
        except Exception as e:
            print(f"Error for alpha {alpha}: {e}")





def ref_reg(alpha, reg_min, reg_max):
    refine_logged_regulairzed_fp(k_0=k_0, k=k,type_3=type_3, max_iter=1,
                                tol=1e-5, integral_mesh_size=12, integral_size=6,
                                lambda_reg_min=reg_min, lambda_reg_max=reg_max,
                                dtype=torch.float64, modified_alpha=alpha)
    refine_logged_fp_tests_regularized(k_0=k_0, k=k,type_3=type_3, R_00=R00, 
                                        lambda_reg_min=reg_min, lambda_reg_max=reg_max,
                                        metric_name='misclassification_test_errors',
                                        modified_alpha=alpha)
    plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=300, n_trials=100,
                                        lambda_reg_max=0.3, lambda_reg_min=0)

 

if __name__ == "__main__":
    
    k,k_0 = 4, 4
    type_3 = 'symmetric'
    R00 = get_R_00(k, type_3)
    d=250
    n_trials=100


    plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=d, n_trials=n_trials,
                                        lambda_reg_max=0.3, lambda_reg_min=0.00)
    sys.exit()

    type_3 = 'two_vs_two_vs_one'
    R00 = get_R_00(k, type_3)

    for _ in range(5): #running 4.80 in three classes close
        refine_logged_fp(k_0=k_0, k=k, type_3=type_3, tol=5*1e-2, use_lambda_reg=0,
                                        max_iter=2, integral_size=4.3, integral_mesh_size=12, 
                                        alpha_max=3.61, alpha_min=3.59, dtype=torch.float64)
        refine_logged_fp_tests(k_0=k_0, k=k, type_3=type_3, metric_name='train_error', alpha_min=3.59, alpha_max=3.61)


        plot_errors_vs_alpha(k=k, k_0=k_0, alpha_min=3, alpha_max=6, d=300, n_trials=100 )












