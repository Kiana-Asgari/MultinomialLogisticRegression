
import numpy as np

from multinomial_logistic.MLE_empirical.ESD_empirical import esd_empirical
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import fit_mle_skitlearn
import matplotlib.pyplot as plt
import os
from multinomial_logistic.log_data.utils import plot_density, plot_regularized_error, plot_errors, plot_regularized_error_vs_d
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
    d=250
    n_trials=30
    R_00 = np.array([[1,0.5], [0.5,1]])
    lambda_reg = 0
    class_err_mean_skitlearn = []
    class_err_std_skitlearn = []

    class_err_mean = []
    class_err_std = []
    for alpha in [2.68]:
        #schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0, k)),
        #                            lambda_reg=1/2*lambda_reg, alpha=alpha, k=k, k_0=k_0, tol=1e-5)
        #A = R_01.T @ sqrtm(np.linalg.inv(R_00))
        #misclass_test_error_theoritical = misclassification_test_error(S, R_00, schur, A, alpha, k, k_0)
        #log_loss_test_error_theoritical = test_error( R_00, schur, R_01, k, k_0, alpha)
        #log_loss_train_error_theoritical = train_error(R_00, schur, R_01, S,alpha, k, k_0)
        #print('theoritical misclassification test error:', misclass_test_error_theoritical  )
        #print('[]theoritical log loss test error:', log_loss_test_error_theoritical)
        #print('[]theoritical log loss train error:', log_loss_train_error_theoritical)  
        results_10 = fit_mle_skitlearn(alpha=alpha, k=k, d=d, n_trials=n_trials,
                                              R_00=R_00)
        print('*****************************alpha=', alpha)
        print('  mle misclassification test error:', np.mean(results_10['misclass_test_errors']), 'std:', np.std(results_10['misclass_test_errors']))
        print('  mle log loss test error:', np.mean(results_10['test_errors']), 'std:', np.std(results_10['test_errors']))
        print('  mle train error:', np.mean(results_10['train_errors']), 'std:', np.std(results_10['train_errors']))
        print('  mle train error:', np.mean(results_10['train_errors']), 'std:', np.std(results_10['train_errors']))
        print('*****************************')


    






   




######################################
from multinomial_logistic.log_data.log_mle_empirical import run_and_log_mle
from multinomial_logistic.MLE_empirical.visualize_data import scatter_plot_data
if __name__ == "__main__":
    print("Running main")
    k = 2
    k_0 = 2
    #test_miscalss()
    #alpha = 3.0
    #d = 250
    run_and_log_fp_tests(k_0=k_0, k=k, non_symmetric=False, two_classes_close=False)

    #run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-4,
    #                max_iter=400, non_symmetric=True, two_classes_close=False)
    #run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-4,
    #                max_iter=400, non_symmetric=False, two_classes_close=False)

    #run_and_log_esd(k_0=k_0, k=k, lambda_reg=0, alpha_input=20.0, R_00_input=np.array([[1,1/2], [1/2,1]]))
    #plot_final_results('density3')

    #run_and_log_fp_classification_test_error(k_0=k_0, k=k, non_symmetric=False)
    #run_and_log_mle(k_0=k_0, k=k, lambda_reg=0, d=250, n_trials=150, non_symmetric=False, two_classes_close=False)
    #R_00 = np.array([[1,1/2], [1/2,1]])
    #plot_regularized_error(k, k_0, R_00, emp_window=0, lambda_reg_max=0.7, lambda_reg_min=0)
    #plot_errors(k_0=k_0, k=k, empirical_window=0.001, alpha_min_emp=3.1, alpha_max=12)

    #run_and_log_fp_tests_regularized(k_0=k_0, k=k,R_00=R_00)
    #run_and_log_fp(k_0=k_0, k=k, lambda_reg=0, tol=1e-2, non_symmetric=False)
    #run_and_log_mle_regularized(k_0=k_0, k=k, lambda_reg=0, d=250, n_trials=100)
    #plot_regularized_error(k, k_0, R_00, emp_window=0.015, lambda_reg_max=0.55)


    #run_and_log_fp_regularized(k=k, k_0=k_0, alpha_values=[10])


    #err = []
    #for alpha in np.linspace(10, 2, 20):
    #    results = fit_mle_skitlearn(alpha=alpha, k=k, d=d, n_trials=5, R_00=R_00_close, verbose=True)
    #    err.append(np.mean(results['misclass_test_errors']))
    #print('err=', err)
    #print('alpha=', np.linspace(10, 2.6, 20))
