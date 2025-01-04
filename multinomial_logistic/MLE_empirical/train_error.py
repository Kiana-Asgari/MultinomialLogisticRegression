import numpy as np
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import log_sum_exp_batch

def train_error(Theta_hat, X, Y_onehot):
    n, d = X.shape
    k = Theta_hat.shape[0]
    Beta_batch_hat = X @ Theta_hat.T
    train_error = 1/n * (log_sum_exp_batch(Beta_batch_hat).sum() - np.sum(Y_onehot * Beta_batch_hat))
    return train_error 