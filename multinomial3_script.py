from configs.config_loader import example_configs
#from multinomial_logistic.log_data.log_fp import run_and_log_fp
#from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from state_evolution.full_recursion import state_evolution_full_recursion
#rom multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests
#from multinomial_logistic.log_data.utils import plot_errors_vs_alpha
import numpy as np
import torch
from configs.R_initiation import get_R_00
if __name__ == "__main__":
    # Load configuration parameters
    print('starting script with GPU')
    #run_and_log_mle(k_0=4, k=4, lambda_reg=0, d=300, n_trials=100, type_3='three_classes_close')



    params = example_configs('k=3_example')
    S, R_01, schur, divergence = state_evolution_full_recursion(*params)
    print(f"  --S: {S}")
    print(f"  --R_01: {R_01}")
    print(f"  --schur: {schur}")
    print('finished script with GPU')
    
    k = 3
    k_0 = 3

    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-5, max_iter=300, type_3='three_classes_close')
    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-5, max_iter=300, type_3='symmetric')
    # run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-5, max_iter=300, type_3='two_classes_close')

   
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3='symmetric')
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3='three_classes_close')
    # run_and_log_fp_tests(k_0=k_0, k=k, lambda_reg=0, type_3='two_classes_close')

    # plot_errors_vs_alpha(k_0=k_0, k=k)