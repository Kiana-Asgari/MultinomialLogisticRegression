
import numpy as np

from state_evolution.full_recursion import state_evolution_full_recursion


from multinomial_logistic.evaluation.log_loss_test_error import  plot_test_error_vs_alpha, plot_test_error_vs_lambda
from multinomial_logistic.evaluation.log_loss_train_eror import plot_train_log_loss_vs_alpha, plot_train_log_loss_vs_lambda
from cubature import cubature
from multinomial_logistic.evaluation.F_norm import plot_norm_vs_lambda_reg
from multinomial_logistic.evaluation.misclassification_test_error import plot_misclass_test_error_vs_lambda, plot_misclass_test_error_vs_alpha

from multinomial_logistic.MLE_empirical.mle_empirical_baseline import  esd_empirical
from multinomial_logistic.ESD_theoretical.Marchenko_Pastur_FP import recover_density, stieltjes_inversion
from multinomial_logistic.ESD_theoretical.ODE import recover_density_via_ODE
from multinomial_logistic.phase_transition.phase_transition import phase_transition_minimization,plot_phase_transition

if __name__ == "__main__":
    print("Running main")
    
    k = 1
    k_0 = 1

    
    #R_00 = np.array([[1,1/2], [1/2,1]])
    R_00 = np.eye(k)
    alpha = 10
    #C_opt, alpha_opt = phase_transition_minimization(R_00=R_00, k=k, k_0=k_0)
    plot_phase_transition(k, k_0)
    #_, empirical_density = esd_empirical(alpha=alpha, k=k, lambda_reg=0, R_00=R_00, max_iter=100, d=250)

    #R_00 = np.eye(k)
    #plot_norm_vs_lambda_reg(R_00, lambda_reg_min=0.2, lambda_reg_max=1.6, k=k, k_0=k_0, max_iter=20,\
    #                         save_path='multinomial_logistic/data/F_norm/3_classes')

    #plot_test_error_vs_lambda(R_00, lambda_reg_min=0.06, lambda_reg_max=1.2, k=k, k_0=k_0, max_iter=25,\
    #                             save_path='multinomial_logistic/data/test_error/3_classes')
    #plot_train_log_loss_vs_lambda(R_00, lambda_reg_min=0.06, lambda_reg_max=1, k=k, k_0=k_0, max_iter=15,\
    #                             save_path='multinomial_logistic/data/train_error/3_classes')
    #MP_iteration(R_00, schur, A, S, z_real, z_imag, alpha, k, k_0)
    #density = recover_density(R_00=R_00, alpha=alpha, k=k, k_0=k_0)
    #print(density)
    #avg_Theta_hat, avg_esd = esd_empirical(alpha=alpha, k=k, lambda_reg=0,\
    #                                       R_00=R_00, d=400, max_iter=10)

    
    #print(type(avg_esd))
    #mle_test()
    #plot_mle_vs_lambda_reg(lambda_reg_min=0.2, lambda_reg_max=0.2, R_00=R_00,\
    #                        k=k, save_path='multinomial_logistic/data/mle/2_classes')



    #R_00 =  4*np.eye(k)
    #plot_test_error_vs_alpha(R_00, alpha_min=5, \
    #                          alpha_max=20, k=k, k_0=k_0, max_iter=15, save_path='multinomial_logistic/data/test_error/2_classes')


    #plot_misclass_test_error_vs_lambda(R_00, lambda_reg_min=0.05, lambda_reg_max=0.8, k=k, k_0=k_0,\
    #                         max_iter=15, save_path='multinomial_logistic/data/misclassification_test_error/2_classes')

    #plot_norm_vs_lambda_reg(R_00, lambda_reg_min=0.1, lambda_reg_max=1.5, k=k, k_0=k_0,\
    #                         max_iter=15, save_path='multinomial_logistic/data/F_norm/3_classes')
    #plot_classification_test_error_vs_alpha(mean_0, variance_0, R_00, alpha_min=10, alpha_max=100,\
    #                          k=k, k_0=k_0, max_iter=20, \
    #                          save_path='multinomial_logistic/data/classification_test_error/4_classes')
    #plot_train_log_loss_vs_alpha(R_00, alpha_min=10, alpha_max=100,\
    #                              k=k, k_0=k_0, max_iter=10, \
   #                               save_path='multinomial_logistic/data/train_error/4_classes')
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

