from configs.config_loader import example_configs
from multinomial_logistic.log_data.log_fp import run_and_log_fp
from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests
from multinomial_logistic.log_data.utils import plot_errors_vs_alpha
import numpy as np
import torch
if __name__ == "__main__":
    # Load configuration parameters
    #params = example_configs('k=3_example')
    #run_and_log_fp(k_0=3, k=3, lambda_reg=0, tol=1e-5, max_iter=300, type_3='three_classes_close')
    #run_and_log_mle(k_0=3, k=3, lambda_reg=0, d=250, n_trials=100, type_3='three_classes_close')
    #run_and_log_fp_tests(k_0=3, k=3, lambda_reg=0, type_3='three_classes_close')
    #plot_errors_vs_alpha(k_0=3, k=3)

    params = example_configs('k=3_example')



    state_evolution_full_recursion(*params)
 

