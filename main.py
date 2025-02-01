import numpy as np
from multinomial_logistic.MLE_empirical.ESD_empirical import esd_empirical
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import fit_mle_skitlearn
import matplotlib.pyplot as plt
import os
from multinomial_logistic.log_data.utils import plot_density, plot_regularized_error, plot_errors_vs_alpha, plot_regularized_error_vs_d
from multinomial_logistic.log_data.log_fp import run_and_log_fp
from multinomial_logistic.log_data.log_esd import run_and_log_esd
from multinomial_logistic.ESD.Marchenko_Pastur_FP import recover_density
from multinomial_logistic.log_data.log_fp_tests import run_and_log_fp_tests_regularized, read_fp_tests_regularized, run_and_log_fp_tests, run_and_log_fp_classification_test_error
from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle_regularized, log_mle_esd, read_mle_esd
from multinomial_logistic.log_data.log_fp_regularized import run_and_log_fp_regularized
from multinomial_logistic.log_data.log_fp import run_and_log_fp



#########################
# final plottings
#########################

def plot_final_results(what_to_plot):
    if what_to_plot == 'regularized_error':
        R_00 = np.array([[1,1/2], [1/2,1]])
        k = 2
        k_0 = 2
        plot_regularized_error(k, k_0, R_00, emp_window=0.015, lambda_reg_max=0.55)

    if what_to_plot == 'density5':
        R_00 = np.array([[1,1/2], [1/2,1]])
        alpha = 5.0
        d = 250
        k = 2
        k_0 = 2
        esd_full_5 = read_mle_esd(k, k_0, alpha=alpha)
        plot_density(k=k, k_0=k_0, R_00=R_00, alpha_target=alpha, eigenvalues=esd_full_5.flatten(), d=d,
                  clean_data_for_3=False, clean_data_for_5=True,\
                  density_lower_bound=1e-2,z_real_lower_bound=0, emp_bins=150)
        
    if what_to_plot == 'density3':
        R_00 = np.array([[1,1/2], [1/2,1]])
        alpha = 3.0
        d = 250
        k = 2
        k_0 = 2
        esd_full_3 = read_mle_esd(k, k_0, alpha=alpha)
        plot_density(k=k, k_0=k_0, R_00=R_00, alpha_target=alpha, eigenvalues=esd_full_3.flatten(), d=d,
                  clean_data_for_3=True, clean_data_for_5=False,clean_data_for_10=False,\
                  density_lower_bound=3*1e-3,z_real_lower_bound=0, emp_bins=150)
        
    if what_to_plot == 'density10':
        R_00 = np.array([[1,1/2], [1/2,1]])
        alpha = 10.0
        d = 250
        k = 2
        k_0 = 2
        esd_full_3 = read_mle_esd(k, k_0, alpha=alpha)
        plot_density(k=k, k_0=k_0, R_00=R_00, alpha_target=alpha, eigenvalues=esd_full_3.flatten(), d=d,
                  clean_data_for_3=False, clean_data_for_5=False, clean_data_for_10=True,\
                  density_lower_bound=2*1e-2,z_real_lower_bound=1e-3,  emp_bins=150)
        
    if what_to_plot == 'density20':
        R_00 = np.array([[1,1/2], [1/2,1]])
        alpha = 20.0
        d = 250
        k = 2
        k_0 = 2
        esd_full_20 = read_mle_esd(k, k_0, alpha=alpha)
        plot_density(k=k, k_0=k_0, R_00=R_00, alpha_target=20.0, eigenvalues=esd_full_20.flatten(), d=d, #fix alpha
                  clean_data_for_3=False, clean_data_for_5=False, clean_data_for_10=False,\
                  clean_data_for_20=True,
                  density_lower_bound=2*1e-2,z_real_lower_bound=1e-3,  emp_bins=150)




from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.evaluation.misclassification_test_error import misclassification_test_error
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from scipy.linalg import sqrtm
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import fit_mle_skitlearn

def test_miscalss():
    k=2
    k_0=2
    d=100
    n_trials=50
    R_00 = np.array([[1,1/2], [1/2,1]])
    lambda_reg = 0
    for alpha in [30]:
        schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0, k)),
                                    lambda_reg=1/2*lambda_reg, alpha=alpha, k=k, k_0=k_0, tol=1e-4)
        A = R_01.T @ (np.linalg.inv(sqrtm(R_00)))
        R_11 = schur + R_01 @ np.linalg.inv(R_00) @ R_01.T

        misclass_test_error_theoritical = misclassification_test_error(S, R_00, schur, A, alpha, k, k_0)
        test_error_theoritical = test_error(R_00, schur, R_01, k, k_0, alpha)
        print('theoritical misclassification test error:', misclass_test_error_theoritical  )
        print('theoritical test error:', test_error_theoritical  )

        #log_loss_test_error_theoritical = test_error( R_00, schur, R_01, k, k_0, alpha)
        #log_loss_train_error_theoritical = train_error(R_00, schur, R_01, S,alpha, k, k_0)
        #print('[]theoritical log loss test error:', log_loss_test_error_theoritical)
        #print('[]theoritical log loss train error:', log_loss_train_error_theoritical)  
        result_skitlearn = fit_mle_skitlearn(alpha=alpha,k=k, d=d, n_trials=n_trials, R_00=R_00)
        _, norms_mle, mle_test_errors, mle_train_errors, mle_misclassification_test_errors = fit_mle_baseline(alpha=alpha,
                                                                                                               k=k, d=d, n_trials=n_trials,
                                                                                                               lambda_reg=lambda_reg, R_00=R_00, return_full_results=True)

        print('*****************************alpha=', alpha)
        print('  mle misclassification test error:', np.mean(mle_misclassification_test_errors), 'std:', np.std(mle_misclassification_test_errors))
        print('  skitlearn misclassification test error:', np.mean(result_skitlearn['misclass_test_errors']), 'std:', np.std(result_skitlearn['misclass_test_errors']))
        print('theoritical misclassification test error:', misclass_test_error_theoritical)
        print('  mle test error:', np.mean(mle_test_errors), 'std:', np.std(mle_test_errors))
        print('  skitlearn test error:', np.mean(result_skitlearn['test_errors']), 'std:', np.std(result_skitlearn['test_errors']))
        print('theoritical test error:', test_error_theoritical)
        print(' mle norms:', np.mean(norms_mle), 'std:', np.std(norms_mle))
        print(' theoritical norms:',np.sqrt(np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T) ) )
        print(' skitlearn norms:', np.mean(result_skitlearn['norms']), 'std:', np.std(result_skitlearn['norms']))



    






   




######################################
from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from multinomial_logistic.MLE_empirical.visualize_data import scatter_plot_data
import argparse

if __name__ == "__main__":
    print("Running main")
    k = 2
    k_0 = 2
    
    # Add argument parsing

    parser = argparse.ArgumentParser(description='Run FP analysis')
    parser.add_argument('--non_symmetric', action='store_true', help='Use non-symmetric configuration')
    parser.add_argument('--two_classes_close', action='store_true', help='Use configuration with two close classes')
    args = parser.parse_args()

    R_00 = np.array([[1,1/2], [1/2,1]])
    #plot_regularized_error(k=k, k_0=k_0, R_00=R_00)

    #run_and_log_fp_tests_regularized(k_0=k_0, k=k, R_00=R_00)  

    #run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-4,
    #               max_iter=400, non_symmetric=args.non_symmetric, 
    #               two_classes_close=args.two_classes_close)
    run_and_log_fp_tests(k_0=k_0, k=k,non_symmetric=args.non_symmetric, 
                        two_classes_close=args.two_classes_close)
    #run_and_log_fp_regularized(k_0=k_0, k=k, alpha_values=[1.5,3,5,10], max_iter=80)





