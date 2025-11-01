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

    R_00 = torch.tensor(params[0])
    schur_0 = torch.tensor(params[1])
    R_01_0 = torch.tensor(params[2])
    lambda_reg = torch.tensor(params[3])
    alpha = torch.tensor(params[4])
    k = torch.tensor(params[5])
    k_0 = torch.tensor(params[6])
    S_0 = torch.tensor(params[7])

    state_evolution_full_recursion(R_00=R_00, schur_0=schur_0, R_01_0=R_01_0, lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, S_0=S_0)
 

