import numpy as np

from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import fit_mle_skitlearn
from multinomial_logistic.utils import batched_mlogit_jacobian



def esd_empirical(alpha, k, lambda_reg, R_00, n_trials = 1, d = 250,
                  X_train=None, X_test=None, y_train=None, y_test=None, skitlearn=False):
    print('************************ESD EMPIRICAL************************')
    print('    ....alpha: ', alpha, 'R_00: ', R_00.flatten())
    if skitlearn:
        results= fit_mle_skitlearn(alpha=alpha, k=k, R_00=R_00, n_trials=n_trials, d=d, 
                                            X_train=X_train, X_test=X_test, y_train_onehot=y_train, y_test_onehot=y_test,
                                            compute_eigenvalues=True)
    else:
        results= fit_mle_baseline(alpha=alpha, k=k, lambda_reg=lambda_reg, R_00=R_00, n_trials=n_trials, d=d, 
                                        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


    return results['eigenvalues']









##########################################
# helper functions
##########################################


