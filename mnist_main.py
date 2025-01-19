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
from mnist_test.log.log_fp_mnist import plot_errors_vs_alpha

if __name__ == "__main__":
    n_hidden = 500
    #plot_errors_vs_alpha(n_hidden=n_hidden, name_data="fashion_mnist", 
    #                          feature_name="tanh", classes_to_keep=[2,4,6])
# Fashion MNIST classes
    
    (x_train, y_train, y_train_one_hot),\
    (x_test, y_test, y_test_one_hot) = get_cleaned_fashion_mnist_data(classes_to_keep=[2,4,6],\
                                                                    normalize=True, pca=False)
    print('avg x norm: ', np.mean(np.linalg.norm(x_train, axis=1)))
    #R_00_relu = learn_mle_on_data(x_train, x_test, y_train, y_test, y_train_one_hot, y_test_one_hot,\
    #                         n_hidden, feature_name='ReLU', data_name='mnist')
    #R_00_rff = learn_mle_on_data(x_train, x_test, y_train, y_test, y_train_one_hot, y_test_one_hot,\
    #                         n_hidden, feature_name='RFF', data_name='mnist')
    R_00_rf, Theta_0, H_train, H_test = learn_mle_on_data(x_train, x_test, y_train, y_test,\
                                                  y_train_one_hot, y_test_one_hot,\
                                                  n_hidden, feature_name='tanh', data_name='fashion_mnist')
    R_00_rf = R_00_rf[1:, 1:]
    Theta_0 = Theta_0[1:, :]
    


    #run_state_evolution_and_save(n_hidden=n_hidden, R_00=R_00_rf, k=2, k_0=2, classes_to_keep=[2,4,6])
    #R_00_none = learn_mle_on_data(x_train, x_test, y_train, y_test, y_train_one_hot, y_test_one_hot,\
    #                        n_hidden, feature_name='none', data_name='mnist')
    #print('irreducible error: ', irreducible_error(R_00_rf, k=2, k_0=2, alpha=None))

    #run_state_evolution_and_save(n_hidden=n_hidden, R_00=R_00, k=k, k_0=k_0)
    
    run_state_evolution_and_save(n_hidden=n_hidden, R_00=R_00_rf, Theta_0=Theta_0, k=2, k_0=2,
                                X_train=H_train, y_train=y_train_one_hot, X_test=H_test, y_test=y_test_one_hot,
                                n_iter=50,  max_iter=200, y_train_full=y_train, y_test_full=y_test,
                                feature_name='tanh', name_data="fashion_mnist",
                                classes_to_keep=[2,4,6])

    
    









