import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from multinomial_logistic.utils import batched_mlogit, log_sum_exp_batch, batched_outer, batched_mlogit_jacobian
from scipy.linalg import sqrtm
from scipy import linalg
from multinomial_logistic.MLE_empirical.utils.test_error import test_error, mle_misclassification_test_error
from multinomial_logistic.MLE_empirical.utils.train_error import train_error
from multinomial_logistic.MLE_empirical.utils.data_generation import generate_data






def negative_log_likelihood_and_gradient(theta, X, Y, lambda_reg):
    n, d = X.shape
    k = theta.shape[0]

    Beta_batch_hat = X @ theta.T
    
    # Add .sum() to ensure we get a scalar value
    nll_regularized = 1/n * (log_sum_exp_batch(Beta_batch_hat).sum() - np.sum(Y * Beta_batch_hat) + \
                     1/2 * lambda_reg * np.linalg.norm(theta)**2)

    probs = batched_mlogit(Beta_batch_hat)[:, :-1] # shape (n,k)
    grad_unregularized = np.einsum('ni,nj->ij', probs, X) - np.einsum('ni,nj->ij', Y, X) # shape (k, d)
    grad = grad_unregularized/n +  lambda_reg * theta
    

    return nll_regularized, grad




def lbfgs_multinomial(X, Y, lambda_reg, 
                      theta_init=None,
                      max_iter=5000, 
                      tol=1e-6,
                      verbose=True):
    """
    Perform L-BFGS optimization to minimize the negative log-likelihood of 
    multinomial logistic regression with L2-regularization.
    
    Arguments:
      X: (n, d) design matrix
      Y: (n, k) one-hot target matrix
      alpha: scaling factor for number of samples vs dimension 
      lambda_reg: regularization parameter
      theta_init: optional initial guess for (k, d) parameters
      max_iter: maximum iterations for L-BFGS
      tol: tolerance for convergence
      verbose: print solver info if True
    
    Returns:
      Theta_hat: (k, d) optimized parameters
      result: the `OptimizeResult` object from scipy
    """
    
    n, d = X.shape
    k = Y.shape[1]

    # If no initial guess is provided, start from random point with norm 1
    if theta_init is None:
        theta_init = np.random.randn(k, d)
        theta_init = theta_init / np.linalg.norm(theta_init)

    # Flatten the (k, d) -> (k*d,)
    theta_init_flat = theta_init.ravel()

    # Define a function for L-BFGS that returns (loss, grad)
    def objective_and_grad(theta_flat):
        # Reshape flat -> (k, d)
        theta = theta_flat.reshape(k, d)
        # Compute negative log-likelihood + grad using your custom function
        nll, grad = negative_log_likelihood_and_gradient(theta, X, Y, lambda_reg)
        # Flatten gradient back to 1D
        grad_flat = grad.ravel()
        return nll, grad_flat

    # Use scipy's L-BFGS-B
    result = minimize(
        fun=objective_and_grad,
        x0=theta_init_flat,
        method='L-BFGS-B',
        jac=True,            # We are supplying the gradient
        options={
            'maxiter': max_iter,
            'disp': verbose,  # Print info if verbose
            'gtol': tol,      # Gradient tolerance
        }
    )

    # Reshape the final solution to (k, d)
    Theta_hat = result.x.reshape(k, d)

    return Theta_hat, result



def fit_mle_baseline(alpha=None, k=None, lambda_reg=0, R_00=None, n_trials = 1, \
                     d = 500, return_full_results = True,\
                     X_train_batch=None, Y_train_batch=None, X_test_batch=None, Y_test_batch=None):

    if X_train_batch is not None:
        n_trials = 1
        d = X_train_batch.shape[1]
        k = Y_train_batch.shape[1]

    zeros_pad = np.zeros((k, d-k))  # k x (d-k) matrix of zeros
    Theta_0 = np.hstack([sqrtm( R_00), zeros_pad]) if X_train_batch is None else np.zeros((k, d)) # concatenate horizontally to get k x d matrix
    Theta_hats = np.zeros((n_trials, k, d))
    norms = np.zeros(n_trials)
    test_errors = np.zeros(n_trials)
    train_errors = np.zeros(n_trials)
    misclassification_test_errors = np.zeros(n_trials)



    for i in range(n_trials):
        if X_train_batch is None:
            X, Y_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0, random_state=2*i)
        else:
            X = X_train_batch
            Y_onehot = Y_train_batch

        Theta_hat, history = lbfgs_multinomial(
            X=X, Y=Y_onehot, lambda_reg=lambda_reg, verbose=False
        )

        Theta_hats[i] = Theta_hat
        norms[i] = np.linalg.norm(Theta_0- Theta_hat)**2

        if X_test_batch is None:
            test_errors[i] = test_error(Theta_0, Theta_hat)
            misclassification_test_errors[i] = mle_misclassification_test_error(Theta_0, Theta_hat)
        else:
            test_errors[i] = train_error(Theta_hat, X_test_batch, Y_test_batch)
            misclassification_test_errors[i] = 0
        train_errors[i] = train_error(Theta_hat, X, Y_onehot)

    avg_Theta_hat = np.mean(Theta_hats, axis=0)
    avg_norms = np.mean(norms)
    avg_test_error = np.mean(test_errors)
    avg_train_error = np.mean(train_errors)
    avg_misclassification_test_error = np.mean(misclassification_test_errors)
    if return_full_results:
        return Theta_hats, norms, test_errors, train_errors, misclassification_test_errors
    else:
        return avg_Theta_hat, avg_norms, avg_test_error, avg_train_error, avg_misclassification_test_error



def esd_empirical(alpha, k, lambda_reg, R_00, max_iter = 100, d = 250, plot_esd = False):
    zeros_pad = np.zeros((k, d-k))
    Theta_0 = np.hstack([sqrtm(R_00), zeros_pad])
    avg_Theta_hat = np.zeros((k, d))
    # Initialize as empty array that we'll append to
    esd_full = np.array([])
    avg_esd = 0
    print(' starting fitting mle with lambda_reg: ', lambda_reg, 'for alpha: ', alpha, 'and k: ', k)
    
    for i in range(max_iter):
        X, Y_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0, random_state=i)
        Theta_hat, history = lbfgs_multinomial(
            X=X, Y=Y_onehot, lambda_reg=lambda_reg, verbose=False
        )

        Hessian = batched_hessian(Theta_hat, X, Y_onehot, alpha, lambda_reg)
        print('norm of X: ', np.mean(np.linalg.norm(X, axis=1)))
        print('Hessian shape: ', Hessian.shape)
        print('first diagonal: ', np.diag(Hessian))
        print('first row: ', Hessian[0,:])
        print('smalleset and largest eigenvalues: ', np.min(np.linalg.eigvalsh(Hessian)), np.max(np.linalg.eigvalsh(Hessian)))
        print('determinant: ', np.linalg.det(Hessian))
        eigenvals = linalg.eigvalsh(Hessian)
        # If this is the first iteration, initialize esd_full with the correct shape
        if i == 0:
            esd_full = np.array([eigenvals])
        else:
            esd_full = np.vstack([esd_full, eigenvals])
        
        # Calculate density values using histogram
        hist, bin_edges = np.histogram(eigenvals, bins=500, density=True)
        max_density = np.max(hist)
        print(f"Iteration {i+1} completed, shape of esd_full: {esd_full.shape}, max density: {max_density:.4f}")
        hist, bin_edges = np.histogram(eigenvals, bins=100, density=True)
        max_density = np.max(hist)
        print(f"2Iteration {i+1} completed, shape of esd_full: {esd_full.shape}, max density: {max_density:.4f}")

    avg_esd = np.mean(esd_full, axis=0)

    if plot_esd:
        # Plot the histogram with transparent fill and black edges
        plt.figure(figsize=(10, 6))
        plt.hist(avg_esd, bins=100, density=True, 
                 facecolor='none', edgecolor='red')
        plt.xlabel('Eigenvalues')
        plt.ylabel('Probability Density')
        plt.title(f'Eigenvalue Spectrum Density (α={alpha}, k={k}, R_00={R_00})')
        
        save_path = f'multinomial_logistic/data/ESD/{k}_classes/esd_alpha_{alpha}_k_{k}_R_00_{R_00}.png'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    return avg_Theta_hat, avg_esd, esd_full




#####################################################
def batched_hessian(theta, X, Y, alpha, lambda_reg):
    n, d = X.shape
    k = theta.shape[0]
    Beta_batch_hat = X @ theta.T
    scale = np.einsum('ni,nj->nij', X, X)
    Hessian = np.zeros((d*k, d*k))
    for i in range(n):
        Hessian += np.kron(batched_mlogit_jacobian(Beta_batch_hat)[i], scale[i])
    return Hessian/n

