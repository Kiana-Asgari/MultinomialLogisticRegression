import numpy as np
import scipy
from real_data.eval.fit_data import fit_data
import matplotlib.pyplot as plt
import os
import json

def eval_errors_empirical(X_train, y_train, X_test, y_test, n_iter,
                          feature_name, n_hidden, file_number=1, seed=42):
     # Create base filename
    base_filename = f"error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "error_empirical", base_filename)
    print('base_filepath', base_filepath)
    
    # Create data/esd directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    print('file path', os.path.dirname(base_filepath))
    
    # Initialize or load existing results
    if os.path.exists(base_filepath):
        print(f"Loading existing file: {os.path.basename(base_filepath)}")
        with open(base_filepath, 'r') as f:
            existing_data = json.load(f)
            results = existing_data["results"]
    else:
        print(f"Creating new file: {os.path.basename(base_filepath)}")
        results = {}
        # Initialize the nested structure
        results = {}

    alpha_values = np.linspace(20, 5, 30)
    for alpha in alpha_values:
        n_samples = int(alpha * X_train.shape[1])
        test_errors = []
        train_errors = []
        classification_errors = []

        for i in range(n_iter):
            np.random.seed(5*i+2)
            sample_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
            X_train_sampled = X_train[sample_indices]
            y_train_sampled = y_train[sample_indices]
            results = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=True, seed=i)
            test_errors.append(results['test_error'])
            train_errors.append(results['train_error'])
            classification_errors.append(results['classification_error'])

        # Calculate mean and std of errors
        results[str(alpha)] = {
            'test_error_mean': float(np.mean(test_errors)),
            'test_error_std': float(np.std(test_errors)),
            'train_error_mean': float(np.mean(train_errors)),
            'train_error_std': float(np.std(train_errors)),
            'classification_error_mean': float(np.mean(classification_errors)),
            'classification_error_std': float(np.std(classification_errors)),
            'n_samples': int(n_samples)
        }
        print('for alpha: ', alpha)
        print('     mean test error: ', results[str(alpha)]['test_error_mean'])
        print('     std test error: ', results[str(alpha)]['test_error_std'])
        print('     mean train error: ', results[str(alpha)]['train_error_mean'])
        print('     std train error: ', results[str(alpha)]['train_error_std'])
        print('mean classification error: ', results[str(alpha)]['classification_error_mean'])
        print('std classification error: ', results[str(alpha)]['classification_error_std'])

        # Save after each alpha computation
        with open(base_filepath, 'w') as f:
            json.dump({
                "metadata": {
                    "feature_name": feature_name,
                    "n_hidden": n_hidden,
                    "n_iter": n_iter,
                    "file_number": file_number
                },
                "results": results
            }, f, indent=2)
        
        print(f"Completed alpha={alpha}, n_samples={n_samples}")
    
    return base_filepath

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



