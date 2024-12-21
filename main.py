from multinomial_logistic.test import test_integration_1, test_integration_2, test_prox, test_fp_integrands, test_fp_system, test_fp_solver
from multinomial_logistic.test import test_fp_solver
from multinomial_logistic.utils import batched_mlogit_jacobian, batched_mlogit
import numpy as np

from state_evolution.S_fp import S_fp_equation, S_fp_solver_new
from state_evolution.full_recursion import S_recursion, R_01_recursion, schur_recursion, integration
from state_evolution.state_evolution_iteration import state_evolution_fixed_point
from state_evolution.full_recursion import state_evolution_full_recursion

from multinomial_logistic.prox import prox_fp_iteration
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import batched_scalar_mult

from multinomial_logistic.evaluation.log_loss_test_error import plot_test_error_vs_lambda_reg, plot_test_error_vs_alpha
from multinomial_logistic.evaluation.log_loss_train_eror import plot_train_log_loss_vs_alpha, plot_train_log_loss_vs_lambda_reg
from cubature import cubature, hcubature


if __name__ == "__main__":
    print("Running main")
    k = 3
    k_0 = 3

    #mean_0 =  np.array([1,1,1])
    #ariance_0 =  np.array([[3,-1,-1], [-1,2,-1],[-1,-1,2]])
    #mean_0 =  4*np.array([1,-1,1])
    #variance_0 =  1/2 *np.array([[1,0,0], [0,1,0], [0,0,1]])
    mean_0 =  np.array( [3,-3,3])
    variance_0 =  np.eye(k)


    R_00 =  variance_0 + np.outer(mean_0, mean_0)
    plot_train_log_loss_vs_alpha(R_00, alpha_min=10, alpha_max=100,\
                                  k=k, k_0=k_0, max_iter=10, \
                                  save_path='multinomial_logistic/data/train_error/4_classes')
    #plot_test_error_vs_alpha(mean_0, variance_0, R_00, alpha_min=15, alpha_max=100,\
    #                           k=k, k_0=k_0, max_iter=10, \
    #                           save_path='multinomial_logistic/data/test_error/4_classes')
    #plot_test_error_vs_lambda_reg(mean_0, variance_0, R_00, lambda_reg_min=0.01, lambda_reg_max=2,\
    #                          alpha=5, k=k, k_0=k_0, max_iter=10, save_path='multinomial_logistic/data/test_error/2_classes')

    #R_01 = np.zeros((k_0,k))
    #R_11 = R_00
    #schur = R_11 - R_01 @ np.linalg.inv(R_00) @ R_01.T


    #Z_theta = np.array([[0,0], [1,1], [2,2]]) #theta_0, theta_1
    #Z_g = np.array([[1,1], [2,2], [0,0]])


  
    
    
    #theta_1, theta_0 = recover_theta(Z_theta, mean_0, variance_0, R_00, schur, R_01, k, k_0)
    #print('theta_1=', theta_1)
    #print('theta_0=', theta_0)
    #g_1, g_0 = recover_g(Z_g, theta_1, theta_0, k, k_0)




    #schur, R_01, S = state_evolution_full_recursion(R_00=R_00, schur_0=schur, R_01_0=R_01,\
    #                                lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
    #
    # 
    # print(R_01)
    #print(test_error(R_00, schur, R_01, alpha, k, k_0))

    #plot_log_loss_vs_lambda_reg(R_00, lambda_reg_min=0, lambda_reg_max=4, alpha=alpha,\
    #                             k=k, k_0=k_0, max_iter=5)
    #plot_log_loss_vs_alpha(R_00=R_00, alpha_min=10, alpha_max=200, lambda_reg=lambda_reg, k=k, k_0=k_0,max_iter=30, 
    #                       save_path='multinomial_logistic/data/log_loss/2_classes')     

    #for lambda_reg in [0.1, 0.5, 1, 2, 5]:
    #    plot_alpha_vs_bias(R_00, alpha_min=1.1, alpha_max=10, lambda_reg=lambda_reg, k=k, k_0=k_0)
    #for alpha in [1.1, 2, 5, 10]:
    #    plot_lambda_vs_bias(R_00, lambda_reg_min=0.1, lambda_reg_max=5, alpha=alpha, k=k, k_0=k_0)
    #plot_alpha_vs_bias(R_00, lambda_reg=0, alpha_min=3.1, alpha_max=10, k=k, k_0=k_0)

