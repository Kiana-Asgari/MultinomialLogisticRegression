import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from mnist_test.log.log_fp_mnist import read_state_evolution_results
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.misclassification_test_error import misclassification_test_error
import numpy as np
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import log_sum_exp_batch
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import batched_hessian
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import os


def error_on_data(Theta_hat, X, Y_onehot):
    n, d = X.shape
    k = Theta_hat.shape[0]
    Beta_batch_hat = X @ Theta_hat.T
    train_error = 1/n * (log_sum_exp_batch(Beta_batch_hat).sum() - np.sum(Y_onehot * Beta_batch_hat))
    return train_error 



def theoretical_test_error(alpha, R_00, k, k_0, tol, lambda_reg):
    schur, R_01, S, divergence = state_evolution_full_recursion(R_00, schur_0=R_00, R_01_0=np.zeros((k,k_0)),\
                                                                  lambda_reg=lambda_reg/2, alpha=alpha, k=k, k_0=k_0, tol=tol)
    
   
    test_error_theoretical = test_error(R_00, schur, R_01, k, k_0, alpha)
    misclass_test_error_theoretical = misclassification_test_error(S, R_00, schur, R_01, alpha, k, k_0)
    return test_error_theoretical.item(), misclass_test_error_theoretical.item()


def esd_empirical(Theta_hat, X_train, Y_train, alpha, lambda_reg):
    Hessian = batched_hessian(Theta_hat, X_train, Y_train, alpha, lambda_reg)
    print('norm of X_train: ', np.mean(np.linalg.norm(X_train, axis=1)))
    print('Hessian shape: ', Hessian.shape)
    print('first diagonal: ', np.diag(Hessian))
    print('first row: ', Hessian[0,:])
    print('smalleset and largest eigenvalues: ', np.min(np.linalg.eigvalsh(Hessian)), np.max(np.linalg.eigvalsh(Hessian)))
    print('determinant: ', np.linalg.det(Hessian))
    eigenvals = np.linalg.eigvalsh(Hessian)
    return eigenvals


def train_and_evaluate_logistic_regression(alpha, R_00, k, k_0, n_hidden, lambda_reg, \
                                           X_train, y_train, X_test, y_test, n_iter=10, tol=1e-2,
                                           y_train_full=None, y_test_full=None, plot_esd=True):
    test_errors = []
    misclass_test_errors = []
    esd_full = None
    # Determine the number of training samples to use
    n_samples = int(alpha * n_hidden)
        
    # Ensure we don't exceed the available training data
    n_samples = min(n_samples, X_train.shape[0])

    print(f"train and evaluate logistic regression with alpha: {alpha}, n_hidden: {n_hidden}, n_iter: {n_iter}")
    print(f" full data size: {X_train.shape[0]}, training on {n_samples} samples")

    for i in range(n_iter):
        # Randomly select n_samples from the training data
        np.random.seed(i)  # Use a different seed each time for variability
        indices = np.random.choice(X_train.shape[0], n_samples, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        y_train_full_subset = y_train_full[indices]
        


        logreg = LogisticRegression(           
                fit_intercept=False,
                penalty=None,
                solver='lbfgs',       # can also use 'sag' or 'saga' if data is large
                max_iter=1000,
        )   

        logreg.fit(X_train_subset, y_train_full_subset)

        Theta_hat = logreg.coef_
        Theta_hat = (Theta_hat - Theta_hat[0])[1:,:]

        avg_test_error = log_loss(y_test_full, logreg.predict_proba(X_test))
        misclass_test_error = 1 - accuracy_score(y_test_full, logreg.predict(X_test))


        test_errors.append(avg_test_error)
        misclass_test_errors.append(misclass_test_error)

    if plot_esd:
        # Plot the histogram with transparent fill and black edges
        print('esd_full shape: ', esd_full.shape)
        plt.figure(figsize=(10, 6))
        # Filter values between 0 and 0.5
        filtered_values = np.clip(np.average(esd_full, axis=0), 0.01, 1)
        plt.hist(filtered_values, bins=200, density=True, 
                    facecolor='none', edgecolor='red')
        plt.xlabel('Eigenvalues')
        plt.ylabel('Probability Density')
        plt.title(f'Eigenvalue Spectrum Density (α={alpha}, k={k}, R_00={R_00})')
        
        save_path = f'mnist_test/log/ESD/esd_alpha_{alpha}RF100PCS.png'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    average_test_error = np.mean(test_errors)
    average_misclass_test_error = np.mean(misclass_test_errors)

    theo_test_error, theo_misclass_test_error = theoretical_test_error(alpha, R_00=R_00, k=k, k_0=k_0, tol=tol, lambda_reg=lambda_reg)
    print(f"theoretical test error: {theo_test_error:.4f}, empirical test error: {average_test_error:.4f}")
    print(f"theoretical misclass test error: {theo_misclass_test_error:.4f}, empirical misclass test error: {average_misclass_test_error:.4f}")
    return average_test_error, theo_test_error, average_misclass_test_error, theo_misclass_test_error
