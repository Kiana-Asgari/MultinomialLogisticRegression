import os

from mnist_test.data_manipulation import get_cleaned_mnist_data, get_cleaned_fashion_mnist_data
from mnist_test.random_feature import learn_mle_on_data
from mnist_test.testing_features_dimension import plot_esd_for_feature
import numpy as np
from scipy.linalg import sqrtm
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.evaluation.misclassification_test_error import misclassification_test_error
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error


import numpy as np
from mnist_test.log.log_fp_mnist import run_state_evolution_and_save
from mnist_test.train_and_eval import train_and_evaluate_logistic_regression
from itertools import combinations
from multinomial_logistic.evaluation.log_loss_test_error import irreducible_error
import seaborn as sns
import matplotlib.pyplot as plt
from mnist_test.log.log_fp_mnist import plot_errors_vs_alpha, edit_theoretical_errors, run_mle_and_save
import sys
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import fit_mle_skitlearn
from mnist_test.testing_features_dimension import _apply_pca
from mnist_test.log.log_fp_mnist import evaluate_mle

"""
R_00 for relu 250 = [[7.50788306 4.3675718 ]
                    [4.3675718  9.05999091]]

R_00 for tanh 250 = [[6.71719346 3.59088112]
                     [3.59088112 8.03657686]]
"""



if __name__ == "__main__":
    R_relu_250 = np.array([[7.50788306, 4.3675718 ],
                          [4.3675718, 9.05999091]])
    R_relu_350 = np.array([[ 8.1515383,   4.74415228],
                          [ 4.74415228, 10.99122296]])

    R_relu_500 = np.array([[14.293405, 8.92488684],
                          [8.92488684, 16.16951883]])

    R_tan_250 = np.array([[6.71719346, 3.59088112],
                          [3.59088112, 8.03657686]])
    R_tan_500 = np.array([[14.22400185, 6.83863084],
                          [6.83863084, 16.28519788]])

    R_tan_350 = np.array([[10.4325381, 5.95547334],
                          [5.95547334, 12.61983397]])
    



    #plot_esd_for_feature(H_train, H_test)




    for n_hidden in [250]:
        n_lower_components = n_hidden
        alphas = [6,6.5,7,7.5,8,9,10,11,12,13,14]

        all_mle_train_errors = []  # List to store mean train errors for each alpha
        all_mle_misclass_test_errors = []  # List to store mean misclass test errors for each alpha
        theoretical_train_errors = []
        theoretical_misclass_test_errors = []
        for alph in alphas:

            (x_train, y_train, y_train_one_hot),\
            (x_test, y_test, y_test_one_hot) = get_cleaned_fashion_mnist_data(classes_to_keep=[2,4,6],\
                                                                            normalize=True, pca=False)


            R_00, Theta_0, H_train, H_test = learn_mle_on_data(x_train, x_test, y_train, y_test,\
                                                    y_train_one_hot, y_test_one_hot,\
                                                    n_hidden, feature_name='tanh+PCA',effective_dim=n_lower_components,
                                                    data_name='fashion_mnist')

            R_00 = R_00[1:,1:]
            Theta_0 = Theta_0[1:,:]
            print('R_00', R_00)
            print('H_train', H_train.shape)
            print('H_test', H_test.shape)
            print('mean each column', np.mean(H_train, axis=0))
            print('std each column', np.std(H_train, axis=0))

            _, mle_train_errors, mle_misclass_test_errors, _ = evaluate_mle(alpha=alph,
                                                            n_hidden=n_hidden,\
                                                            R_00=R_00, Theta_0=Theta_0, k=2, k_0=2, \
                                                            X_train=H_train, y_train=y_train, X_test=H_test, y_test=y_test, \
                                                            n_iter=50, tol=1e-4, y_train_full=y_train, y_test_full=y_test, plot_esd=False)
            print('mle misclass error', np.mean(mle_misclass_test_errors))
            print('mle train error', np.mean(mle_train_errors))
            all_mle_train_errors.append(np.mean(mle_train_errors))
            all_mle_misclass_test_errors.append(np.mean(mle_misclass_test_errors))

            # Run the state evolution recursion
            schur_0, R_01_0, S_0, divergence = state_evolution_full_recursion(
                                R_00=R_00, schur_0=R_00, R_01_0=np.zeros((2,2)),
                                alpha=alph, k=2, k_0=2, lambda_reg=0, S_0=np.eye(2),
                                tol=1e-3, max_iter=30
            )
            A = R_01_0 @ np.linalg.inv(sqrtm(R_00))
            misclass_test_error_theoretical = misclassification_test_error(S=None, R_00=R_00, schur_t=schur_0, A_t=A, alpha=alph, k=2, k_0=2)
            train_error_theoretical = train_error(R_00=R_00, schur=schur_0, R_01=R_01_0, S=S_0, alpha=alph, k=2, k_0=2)
            print('for dim ', n_hidden, ' theoretical train error', train_error_theoretical)
            print('for  dim ', n_hidden, ' mle train error', np.mean(mle_train_errors))
            print('for  dim ', n_hidden, ' theoretical misclass error', misclass_test_error_theoretical)
            print('for  dim ', n_hidden, ' mle misclass error', np.mean(mle_misclass_test_errors))
            theoretical_train_errors.append(train_error_theoretical)
            theoretical_misclass_test_errors.append(misclass_test_error_theoretical)

        

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, all_mle_train_errors, 'o', label='MLE Train Error')
        plt.plot(alphas, theoretical_train_errors, '-', label='Theoretical Train Error')
        plt.xlabel('Alpha')
        plt.ylabel('Train Error')
        plt.title('MLE vs Theoretical Train Error')
        plt.legend()
        plt.grid(True)
        
        # Create directory if it doesn't exist
        save_dir = os.path.join("mnist_test", "log", "figures", "pcaappliednew")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save plot
        save_path = os.path.join(save_dir, f"tanh_train_errors_nhidden_{n_hidden}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        #misclass

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, all_mle_misclass_test_errors, 'o', label='MLE Misclassification Test Error')
        plt.plot(alphas, theoretical_misclass_test_errors, '-', label='Theoretical Misclassification Test Error')
        plt.xlabel('Alpha')
        plt.ylabel('Misclassification Test Error')
        plt.title('MLE vs Theoretical Misclassification Test Error')
        plt.legend()
        plt.grid(True)
        
        # Create directory if it doesn't exist
        save_dir = os.path.join("mnist_test", "log", "figures", "pcaappliednew")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save plot
        save_path = os.path.join(save_dir, f"tanh_misclass_errors_nhidden_{n_hidden}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    sys.exit()
    run_mle_and_save(n_hidden=n_hidden, R_00=R_00, Theta_0=Theta_0, k=2, k_0=2,
                                X_train=H_train, y_train=y_train_one_hot, X_test=H_test, y_test=y_test_one_hot,
                                n_iter=100,  max_iter=100, y_train_full=y_train, y_test_full=y_test,
                                feature_name=feature_name, name_data="fashion_mnist",
                                classes_to_keep=[2,4,6])





    
    









