from multinomial_logistic.test import test_integration_1, test_integration_2, test_prox, test_fp_integrands, test_fp_system, test_fp_solver
from multinomial_logistic.test import test_fp_solver
from multinomial_logistic.utils import batched_mlogit_jacobian, batched_mlogit
import numpy as np

from state_evolution.S_fp import S_fp_equation, S_fp_solver_new
from state_evolution.full_recursion import S_recursion, R_01_recursion, schur_recursion
from state_evolution.state_evolution_iteration import state_evolution_fixed_point
from state_evolution.full_recursion import state_evolution_full_recursion
from test_error.utils import plot_array
from test_error.bias import plot_alpha_vs_bias, plot_lambda_vs_bias
from test_error.log_loss import plot_log_loss_vs_lambda_reg, plot_log_loss_vs_alpha
if __name__ == "__main__":
    print("Running main")
    k = 3
    k_0 = 3
   
    #R_00 = 1/2 * np.array(np.array([[2,1,1],
    #                             [1,2,1],
    #                             [1,1,2]]))
    R_00 = np.eye(k_0)
    R_01 = np.zeros((k_0,k))
    R_11 =  R_00
    schur = R_11 - R_01 @ np.linalg.inv(R_00) @ R_01.T
    print(f'     schur: {schur}')
    #R_00 =  np.eye(k_0)

    alpha = 30
    lambda_reg = 0

    #state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
    schur, R_01, S = state_evolution_full_recursion(R_00=R_00, schur_0=schur, R_01_0=R_01,\
                                    lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)

    #plot_log_loss_vs_lambda_reg(R_00, lambda_reg_min=0.1, lambda_reg_max=5, alpha=alpha, k=k, k_0=k_0)
    #plot_log_loss_vs_alpha(R_00, alpha_min=1, alpha_max=15, lambda_reg=lambda_reg, k=k, k_0=k_0)     

    #for lambda_reg in [0.1, 0.5, 1, 2, 5]:
    #    plot_alpha_vs_bias(R_00, alpha_min=1.1, alpha_max=10, lambda_reg=lambda_reg, k=k, k_0=k_0)
    #for alpha in [1.1, 2, 5, 10]:
    #    plot_lambda_vs_bias(R_00, lambda_reg_min=0.1, lambda_reg_max=5, alpha=alpha, k=k, k_0=k_0)
    #plot_alpha_vs_bias(R_00, lambda_reg=0, alpha_min=3.1, alpha_max=10, k=k, k_0=k_0)

