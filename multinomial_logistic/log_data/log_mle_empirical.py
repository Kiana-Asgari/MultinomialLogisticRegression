# This file is used to log the results of the MLE empirical experiments
# The results are saved in the log_data/data/mle_empirical folder
# The results are saved in the following format:
# - alpha, k, lambda_reg, R_00, n_trials, d, return_full_results
# - The results are saved in the following format:
# for each alpha, k, lambda_reg, R_00, n_trials, d
# - norm : array of size n_trials that stores ||theta_hat - theta_0||_F 
# - test_errors : array of size n_trials that stores all the test errors
# - train_errors : array of size n_trials that stores all the train errors
# - avg_theta_hat : array of size n_trials that stores all the theta_hat

# For each alpha, k, lambda_reg, R_00,, the read function returns the data found for the closest logged alpha

import numpy as np
import json
import os
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline
import warnings

def run_and_log_mle(k_0, k, lambda_reg=0):     
    # Set parameters
    n_trials = 100
    d = 250
    
    # Create base filename without timestamp, but with d and n_trials
    base_filename = f"mle_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "mle_empirical", base_filename)
    
    # Create data/mle_empirical directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Check for existing files with matching parameters
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        if f"mle_data_lambda{lambda_reg}_d{d}_ntrials{n_trials}" in filename:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                existing_files.append((filename, data))
    
    # If matching file exists, use it
    if existing_files:
        filename, existing_data = existing_files[0]
        filepath = os.path.join(data_dir, filename)
        results = existing_data["results"]
        print(f"Appending to existing file: {filename}")
    else:
        # Create new file
        filepath = base_filepath
        results = {}
        print(f"Creating new file: {os.path.basename(filepath)}")

    R_00_values = np.array([np.eye(k), [[1,0.5], [0.5,1]]])  
    alphas = np.concatenate([np.linspace(2.8, 5, 20), np.linspace(5, 20, 30)]).flatten()
    alphas = np.sort(alphas)[::-1]  # Sort in decreasing order
    n_trials = 100
    d = 250

    for R_00 in R_00_values:
        for alpha in alphas:
            alpha_str = str(alpha)
            R_00_str = str(R_00.tolist())
            
            # Skip if we already have results for this alpha and R_00
            if R_00_str in results and alpha_str in results[R_00_str]:
                print(f"Skipping alpha={alpha} for R_00={R_00} (already exists)")
                continue
                
            print(f"\nRunning alpha = {alpha}")
            
            # Set up warning catching
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    Theta_hats, norms, test_errors, train_errors = fit_mle_baseline(
                        alpha=alpha,
                        k=k,
                        lambda_reg=lambda_reg,
                        R_00=R_00,
                        n_trials=n_trials,
                        d=d,
                        return_full_results=True
                    )
                    
                    # Check if overflow warning was raised
                    if any(issubclass(warn.category, RuntimeWarning) and "overflow" in str(warn.message) for warn in w):
                        raise FloatingPointError("Overflow detected in computation")
                    
                    # Check if average norm exceeds threshold
                    avg_norm = np.mean(norms)
                    diverged = avg_norm > 1e3
                    
                    # Initialize R_00 dict if it doesn't exist
                    if R_00_str not in results:
                        results[R_00_str] = {}
                    
                    if diverged:
                        # If diverged due to large norm, save empty arrays
                        results[R_00_str][alpha_str] = {
                            "norm": [],
                            "test_errors": [],
                            "train_errors": [],
                            "theta_hats": [],
                            "shapes": {
                                "theta_hats": [0, k, d],
                                "norms": [0],
                                "test_errors": [0],
                                "train_errors": [0]
                            },
                            "diverged": True
                        }
                        print(f"Diverged due to large norm (avg_norm={avg_norm:.2f}) for alpha={alpha}")
                    else:
                        results[R_00_str][alpha_str] = {
                            "norm": norms.tolist(),
                            "test_errors": test_errors.tolist(),
                            "train_errors": train_errors.tolist(),
                            "theta_hats": Theta_hats.tolist(),
                            "shapes": {
                                "theta_hats": list(Theta_hats.shape),
                                "norms": list(norms.shape),
                                "test_errors": list(test_errors.shape),
                                "train_errors": list(train_errors.shape)
                            },
                            "diverged": False
                        }
                
                except (FloatingPointError, RuntimeError) as e:
                    print(f"Numerical error detected for alpha={alpha}: {str(e)}")
                    if R_00_str not in results:
                        results[R_00_str] = {}
                    
                    results[R_00_str][alpha_str] = {
                        "norm": [],
                        "test_errors": [],
                        "train_errors": [],
                        "theta_hats": [],
                        "shapes": {
                            "theta_hats": [0, k, d],
                            "norms": [0],
                            "test_errors": [0],
                            "train_errors": [0]
                        },
                        "diverged": True
                    }
            
            # Save results after each alpha without pretty printing and with compact arrays
            with open(filepath, 'w') as f:
                json.dump({
                    "metadata": {
                        "k": k,
                        "k_0": k_0,
                        "lambda_reg": lambda_reg,
                        "n_trials": n_trials,
                        "d": d
                    },
                    "results": results
                }, f, separators=(',', ':'))  # Most compact format
            
    return filepath

def read_mle_results(alpha, k, k_0, R_00, lambda_reg=0, d=250, n_trials=100):
    """
    Read results for the closest available alpha value from the matching file.
    
    Args:
        alpha (float): The alpha value to look for
        k (int): Dimension of the system
        k_0 (int): Dimension of g_0
        R_00 (ndarray): Initial covariance matrix
        lambda_reg (float): Regularization parameter
        d (int): Dimension of the problem
        n_trials (int): Number of trials
    
    Returns:
        tuple: (norm, test_errors, train_errors, theta_hats, diverged) if found, None if not found
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "mle_empirical")
    
    if not os.path.exists(data_dir):
        print("No data directory found")
        return None
    
    # Look for the specific file with d and n_trials in the name
    filename = f"mle_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}.json"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"No file found matching parameters k={k}, k_0={k_0}, lambda={lambda_reg}")
        return None
    
    # Read the file
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Check if we have results for this R_00
    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        print(f"No results found for R_00={R_00}")
        return None
    
    # Get all available alphas for this R_00
    available_alphas = [float(a) for a in data["results"][R_00_str].keys()]
    
    if not available_alphas:
        print(f"No valid results found for R_00={R_00}")
        return None
    
    # Find closest alpha
    available_alphas = np.array(available_alphas)
    closest_alpha = available_alphas[np.argmin(np.abs(available_alphas - alpha))]
    closest_alpha_str = str(closest_alpha)
    
    result = data["results"][R_00_str][closest_alpha_str]
    
    # First check if the computation diverged
    if result["diverged"]:
        print(f"Warning: Computation diverged for alpha={closest_alpha}")
        return None, None, None, None, True
    
    # Convert lists back to numpy arrays with proper shapes
    shapes = result["shapes"]
    norm = np.array(result["norm"]).reshape(shapes["norms"])
    test_errors = np.array(result["test_errors"]).reshape(shapes["test_errors"])
    train_errors = np.array(result["train_errors"]).reshape(shapes["train_errors"])
    theta_hats = np.array(result["theta_hats"]).reshape(shapes["theta_hats"])
    
    print(f"Found results in file: {filename}")
    print(f"Using alpha={closest_alpha} (requested alpha={alpha})")
    return norm, test_errors, train_errors, theta_hats, False
