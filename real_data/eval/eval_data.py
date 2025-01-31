import numpy as np
import scipy
from real_data.eval.fit_data import fit_data
import matplotlib.pyplot as plt
import os
import json





def eval_bayesian_error(X_train, y_train, X_test, y_test, feature_name, n_hidden, file_number=1, seed=42):

    # Create base filename
    base_filename = f"bayesian_error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
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



    n_iter = 100
    test_errors = np.zeros(n_iter)
    train_errors = np.zeros(n_iter)
    classification_errors = np.zeros(n_iter)


    for i in range(n_iter):
        mask = np.ones(len(X_train), dtype=bool)
        mask[2*i] = False
        X_train_sampled = X_train[mask]
        y_train_sampled = y_train[mask]
        results_iter = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=False, seed=i)
        test_errors[i] = float(results_iter['test_error'])
        train_errors[i] = float(results_iter['train_error'])
        classification_errors[i] = float(results_iter['classification_error'])
        print('iter: ', i, 'test error: ', test_errors[i], 'train error: ', train_errors[i], 'classification error: ', classification_errors[i])

    # Store results for this alpha
    results = {
        "test_errors": test_errors.tolist(),
        "train_errors": train_errors.tolist(),
        "classification_errors": classification_errors.tolist(),
        "mean_test_error": float(np.mean(test_errors)),
        "mean_train_error": float(np.mean(train_errors)),
        "mean_classification_error": float(np.mean(classification_errors)),
        "std_test_error": float(np.std(test_errors)),
        "std_train_error": float(np.std(train_errors)),
        "std_classification_error": float(np.std(classification_errors))
    }

    # Save results after each alpha computation
    data_to_save = {
        "feature_name": feature_name,
        "n_hidden": n_hidden,
        "n_iter": n_iter,
        "results": results
    }
    
    with open(base_filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print('     mean test error: ', np.mean(test_errors), 'std test error: ', np.std(test_errors))
    print('     mean train error: ', np.mean(train_errors), 'std train error: ', np.std(train_errors))
    print('     mean classification error: ', np.mean(classification_errors), 'std classification error: ', np.std(classification_errors))

    return results






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

    alpha_values = np.linspace(4.5, 20, 35)
    for alpha in alpha_values:
        alpha_str = str(float(alpha))  # Convert to float first to ensure proper string conversion
        n_samples = int(alpha * X_train.shape[1])
        test_errors = np.zeros(n_iter)
        train_errors = np.zeros(n_iter)
        classification_errors = np.zeros(n_iter)

        # Skip computation if we already have results for this alpha
        if alpha_str in results:
            print(f"Skipping alpha={alpha_str} (already computed)")
            continue

        for i in range(n_iter):
            np.random.seed(5*i+2)
            sample_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
            X_train_sampled = X_train[sample_indices]
            y_train_sampled = y_train[sample_indices]
            results_iter = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=False, seed=i)
            test_errors[i] = float(results_iter['test_error'])
            train_errors[i] = float(results_iter['train_error'])
            classification_errors[i] = float(results_iter['classification_error'])
            print('iter: ', i, 'test error: ', test_errors[i], 'train error: ', train_errors[i], 'classification error: ', classification_errors[i])

        # Store results for this alpha
        results[alpha_str] = {
            "test_errors": test_errors.tolist(),
            "train_errors": train_errors.tolist(),
            "classification_errors": classification_errors.tolist(),
            "mean_test_error": float(np.mean(test_errors)),
            "mean_train_error": float(np.mean(train_errors)),
            "mean_classification_error": float(np.mean(classification_errors)),
            "std_test_error": float(np.std(test_errors)),
            "std_train_error": float(np.std(train_errors)),
            "std_classification_error": float(np.std(classification_errors))
        }

        # Save results after each alpha computation
        data_to_save = {
            "feature_name": feature_name,
            "n_hidden": n_hidden,
            "n_iter": n_iter,
            "results": results
        }
        
        with open(base_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        
        print(f"Saved results for alpha={alpha_str}")
        print('     mean test error: ', np.mean(test_errors), 'std test error: ', np.std(test_errors))
        print('     mean train error: ', np.mean(train_errors), 'std train error: ', np.std(train_errors))
        print('     mean classification error: ', np.mean(classification_errors), 'std classification error: ', np.std(classification_errors))

    return results

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



