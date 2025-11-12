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


 

# whats left from regularized fp: wait for the soal11 run to finish. It is modifiying 1.5, 3, 5 alphas.
# Then, merge the fp_solution with fp_solution(old).json; rename the old file to be tha main file;
# change the @log_fp_regularized to use the new file not the old one.
# run the fp_tests_regularized with the new file; currently it only has half of the 10 alpha.
# plot the results.



# Whats left from fp: wait for the soal10 run to finish. It is midofiying symmetric fp.
# After, look at the plots of symmetric fp. Then, you need to keo refining two classes close fp.
#important: On soal 10; only use two fences here. The full fences are for soal 11.
if __name__ == "__main__":
    k,k_0 = 4, 4
    R00 = get_R_00(k, "symmetric")  

    for type_3 in ['symmetric']: #need to run this
        R00 = get_R_00(k, type_3)
        refine_logged_fp(k_0=k_0, k=k, type_3=type_3, tol=1e-5, use_lambda_reg=0,
                                        max_iter=4, integral_size=5, integral_mesh_size=13, 
                                        alpha_max=5, alpha_min=3.6, dtype=torch.float32)

        refine_logged_fp_tests(k_0=k_0, k=k, type_3=type_3, metric_name='train_error')
        plot_errors_vs_alpha(k=k, k_0=k_0, alpha_min=3.6, alpha_max=14.0, d=300)
    sys.exit()
    lambdas = np.linspace(0.05, 0.3, 20)

    run_and_log_fp_regularized(k_0=k_0, k=k, type_3=type_3, alphas=np.array([10]), lambda_regs=lambdas,
                             dtype=torch.float32, integral_mesh_size=13, integral_size=5, max_iter=30, tol=1e-4)
    run_and_log_fp_tests_regularized(k_0=k_0, k=k,type_3=type_3, R_00=R00)
    plot_regularized_error(k=k, k_0=k_0, R_00=R00, type_3=type_3, d=300, n_trials=100, lambda_reg_max=0.3, lambda_reg_min=0)










