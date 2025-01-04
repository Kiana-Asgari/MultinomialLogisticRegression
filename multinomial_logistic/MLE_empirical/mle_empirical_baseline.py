import numpy as np
import os

from scipy.optimize import minimize

from multinomial_logistic.utils import batched_mlogit, log_sum_exp_batch, batched_outer, batched_mlogit_jacobian
from scipy.linalg import sqrtm
from scipy import linalg
from multinomial_logistic.MLE_empirical.test_error import test_error
from multinomial_logistic.MLE_empirical.train_error import train_error

def generate_data(alpha, d, k, Theta_0, random_state=0):
    np.random.seed(random_state)
    n = np.ceil(alpha*d).astype(int)
    X = np.random.randn(n, d)  # shape (n, d)
    
    # Logits for non-baseline classes: shape (n, k)
    Beta_batch = X @ Theta_0.T
    prob_y_batch = batched_mlogit(Beta_batch)
    cdf_y_batch = np.cumsum(prob_y_batch, axis=1)

    # Prob(class j) for j=1..k    
    # Draw labels
    Y_onehot = np.zeros((n, k))  # shape (n, k)
    random_values = np.random.uniform(0, 1, size=(n,))
    samples = np.argmax(random_values[:, None] <= cdf_y_batch, axis=1) # e_j => label j-1, 0 => label k

    # convert to one-hot
    Y_onehot = np.zeros((n, k))  # shape (n, k)
    # Only set 1.0 for non-baseline classes (when samples < k)
    baseline_mask = samples < k
    Y_onehot[np.arange(n)[baseline_mask], samples[baseline_mask]] = 1.0

    return X, Y_onehot





def negative_log_likelihood_and_gradient(theta, X, Y, alpha, lambda_reg):
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




def lbfgs_multinomial(X, Y, alpha, lambda_reg, 
                      theta_init=None,
                      max_iter=1000, 
                      tol=1e-5,
                      verbose=False):
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
        nll, grad = negative_log_likelihood_and_gradient(theta, X, Y, alpha, lambda_reg)
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



def fit_mle_baseline(alpha, k, lambda_reg, R_00, n_trials, d = 500):

    zeros_pad = np.zeros((k, d-k))  # k x (d-k) matrix of zeros
    Theta_0 = np.hstack([sqrtm( R_00), zeros_pad])  # concatenate horizontally to get k x d matrix
    avg_Theta_hat = np.zeros((k, d))
    avg_norm = 0
    avg_test_error = 0
    avg_train_error = 0
    print(' starting fitting mle with lambda_reg: ', lambda_reg, 'for alpha: ', alpha, 'and k: ', k)

    for i in range(n_trials):
        X, Y_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0, random_state=i)
        Theta_hat, history = lbfgs_multinomial(
            X=X, Y=Y_onehot, alpha=alpha, lambda_reg=lambda_reg, verbose=False
        )

        avg_norm += np.linalg.norm(Theta_0- Theta_hat)**2
        avg_Theta_hat += Theta_hat
        avg_test_error += test_error(Theta_0, Theta_hat)
        avg_train_error += train_error(Theta_hat, X, Y_onehot)

    avg_Theta_hat /= n_trials
    avg_norm /= n_trials
    avg_test_error /= n_trials
    avg_train_error /= n_trials
    return avg_Theta_hat, avg_norm, avg_test_error, avg_train_error

import matplotlib.pyplot as plt


def esd_empirical(alpha, k, lambda_reg, R_00, max_iter = 100, d = 250):

    zeros_pad = np.zeros((k, d-k))  # k x (d-k) matrix of zeros
    Theta_0 = np.hstack([sqrtm( R_00), zeros_pad])  # concatenate horizontally to get k x d matrix
    avg_Theta_hat = np.zeros((k, d))
    avg_esd = 0
    print(' starting fitting mle with lambda_reg: ', lambda_reg, 'for alpha: ', alpha, 'and k: ', k)
    for i in range(max_iter):
        X, Y_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0, random_state=i)
        Theta_hat, history = lbfgs_multinomial(
            X=X, Y=Y_onehot, alpha=alpha, lambda_reg=lambda_reg, verbose=False
        )

        Hessian = batched_hessian(Theta_hat, X, Y_onehot, alpha, lambda_reg)
        avg_esd += linalg.eigvalsh(Hessian)

    avg_esd /= max_iter

    # Plot the histogram with transparent fill and black edges
    plt.figure(figsize=(10, 6))
    plt.hist(avg_esd, bins=50, density=True, 
             facecolor='none', edgecolor='red')
    plt.xlabel('Eigenvalues')
    plt.ylabel('Probability Density')
    plt.title(f'Eigenvalue Spectrum Density (α={alpha}, k={k}, R_00={R_00})')
    
    save_path = f'multinomial_logistic/data/ESD/{k}_classes/esd_alpha_{alpha}_k_{k}_R_00_{R_00}.png'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


    return avg_Theta_hat, avg_esd




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

