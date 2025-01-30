import numpy as np
import scipy
from real_data.eval.fit_data import fit_data
import matplotlib.pyplot as plt
import os


def eval_esd_hessian(X_train, y_train, X_test, y_test, alpha, n_iter, seed=42):
    
    n_samples = alpha * X_train.shape[1]
    esd_values = []

    for i in range(n_iter):
        np.random.seed(5*i+2)
        sample_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train_sampled = X_train[sample_indices]
        y_train_sampled = y_train[sample_indices]
        results = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=True, seed=i)
        esd_values.append(results['esd_values'])

    print("esd_values min: ", np.min(esd_values), "max: ", np.max(esd_values))
        
    # Create histogram and plot MP distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(esd_values).flatten(), bins=100, density=True, alpha=0.7,
             color='blue', label=f'Empirical for $\\alpha={alpha}$')


    # Add labels and title
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title(f'ESD of Hessian')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create directory if it doesn't exist
    save_dir = os.path.join("real_data", "eval", "log", "ESD")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(save_dir, f"esd_hessian_alpha_{alpha}.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"ESD plot saved to {save_path}")

    return esd_values

    





def eval_data(X_train, y_train, X_test, y_test, alpha, n_iter):

    n_samples = alpha * X_train.shape[1]
    print("n_samples: ", n_samples)
    test_errors = []
    train_errors = []
    classification_errors = []

    for i in range(n_iter):
        np.random.seed(5*i+2)
        sample_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train_sampled = X_train[sample_indices]
        y_train_sampled = y_train[sample_indices]
        results = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=False, seed=i)
        test_errors.append(results['test_error'])
        train_errors.append(results['train_error'])
        classification_errors.append(results['classification_error'])

    return test_errors, train_errors, classification_errors



