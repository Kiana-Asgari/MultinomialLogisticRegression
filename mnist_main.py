import os

from mnist_test.data_manipulation import get_cleaned_mnist_data, get_cleaned_fashion_mnist_data
from mnist_test.random_feature import learn_mle_on_data
import numpy as np


import numpy as np
from mnist_test.log.log_fp_mnist import run_state_evolution_and_save
from mnist_test.train_and_eval import train_and_evaluate_logistic_regression
from itertools import combinations
from multinomial_logistic.evaluation.log_loss_test_error import irreducible_error
import seaborn as sns
import matplotlib.pyplot as plt
from mnist_test.log.log_fp_mnist import plot_errors_vs_alpha, edit_theoretical_errors, run_mle_and_save
import sys


"""
R_00 for relu 250 = [[7.50788306 4.3675718 ]
                    [4.3675718  9.05999091]]

R_00 for tanh 250 = [[6.71719346 3.59088112]
                     [3.59088112 8.03657686]]
"""



if __name__ == "__main__":
    n_hidden = 350
    R_relu_250 = np.array([[7.50788306, 4.3675718 ],
                          [4.3675718, 9.05999091]])
    R_relu_500 = np.array([[14.293405, 8.92488684],
                          [8.92488684, 16.16951883]])

    R_tan_250 = np.array([[6.71719346, 3.59088112],
                          [3.59088112, 8.03657686]])
    R_tan_500 = np.array([[14.22400185, 6.83863084],
                          [6.83863084, 16.28519788]])

    R_tan_350 = np.array([[10.4325381, 5.95547334],
                          [5.95547334, 12.61983397]])


    #run_state_evolution_and_save(n_hidden=n_hidden, R_00=R_tan_350, k=2, k_0=2,
    #                            feature_name='tanh', name_data="fashion_mnist", max_iter=100,
    #                            classes_to_keep=[2,4,6])
    #sys.exit()
    #plot_errors_vs_alpha(n_hidden=n_hidden, name_data="fashion_mnist", 
    #                          feature_name="tanh", classes_to_keep=[2,4,6])
    #sys.exit()
    #edit_theoretical_errors(R_00=R_relu_250, k=2, k_0=2, n_hidden=n_hidden, feature_name='ReLU',
    #                         name_data='fashion_mnist', max_iter=50)


    #sys.exit()
    #now doing relu

    (x_train, y_train, y_train_one_hot),\
    (x_test, y_test, y_test_one_hot) = get_cleaned_fashion_mnist_data(classes_to_keep=[2,4,6],\
                                                                    normalize=True, pca=False)
    print('avg x norm: ', np.mean(np.linalg.norm(x_train, axis=1)))

    
    R_00_rf, Theta_0, H_train, H_test = learn_mle_on_data(x_train, x_test, y_train, y_test,\
                                                  y_train_one_hot, y_test_one_hot,\
                                                  n_hidden, feature_name='tanh', data_name='fashion_mnist')
    R_00_rf = R_00_rf[1:, 1:]
    Theta_0 = Theta_0[1:, :]



    run_mle_and_save(n_hidden=n_hidden, R_00=R_tan_350, Theta_0=Theta_0, k=2, k_0=2,
                                X_train=H_train, y_train=y_train_one_hot, X_test=H_test, y_test=y_test_one_hot,
                                n_iter=100,  max_iter=100, y_train_full=y_train, y_test_full=y_test,
                                feature_name='tanh', name_data="fashion_mnist",
                                classes_to_keep=[2,4,6])

    



    
    









