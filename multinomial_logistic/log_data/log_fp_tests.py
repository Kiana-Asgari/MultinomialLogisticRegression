# logging the theoretical test errors, train errors, F_norm 
# using the results from the fp_solution folder
# for different values of alpha, k, k_0, R_00

import numpy as np
import json
import os
from multinomial_logistic.log_data.log_fp import read_fp_results
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error

def run_and_log_fp_tests(k_0, k, lambda_reg=0):     
    # Create base filename
    base_filename = f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "fp_tests", base_filename)
    
    # Create data/fp_tests directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Check for existing files with matching parameters
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        if f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}" in filename:
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

    # Read available alphas and R_00 values from FP solution files
    fp_data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_solution")
    fp_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    fp_filepath = os.path.join(fp_data_dir, fp_filename)
    
    if not os.path.exists(fp_filepath):
        print(f"No FP solution file found: {fp_filename}")
        return None
        
    with open(fp_filepath, 'r') as f:
        fp_data = json.load(f)
        
    # Extract unique R_00 values and alphas from FP solutions
    R_00_values = []
    alphas = set()
    
    for R_00_str in fp_data["results"].keys():
        R_00 = np.array(json.loads(R_00_str))
        R_00_values.append(R_00)
        alphas.update(float(alpha) for alpha in fp_data["results"][R_00_str].keys())
    
    alphas = sorted(list(alphas), reverse=True)  # Sort in decreasing order
    
    for R_00 in R_00_values:
        for _alpha in alphas:
            alpha_str = str(_alpha)
            R_00_str = str(R_00.tolist())
            
            # Skip if we already have results for this alpha and R_00
            if R_00_str in results and alpha_str in results[R_00_str]:
                print(f"Skipping alpha={_alpha} for R_00={R_00} (already exists)")
                continue
                
            print(f"\nProcessing alpha = {_alpha}")
            
            # Get FP solution results
            fp_results = read_fp_results(alpha=_alpha, k=k, k_0=k_0, R_00=R_00, lambda_reg=lambda_reg)
            if fp_results is None:
                print(f"No FP solution found for alpha={_alpha}")
                continue
                
            schur, R_01, S, diverged, closest_alpha = fp_results
            
            if diverged:
                # If solution diverged, store empty results
                if R_00_str not in results:
                    results[R_00_str] = {}
                    
                results[R_00_str][alpha_str] = {
                    "test_error": None,
                    "train_error": None,
                    "F_norm": None,
                    "diverged": True,
                    "actual_alpha": closest_alpha
                }
                continue

            # Calculate test error, train error, and F_norm
            R_11 = schur + R_01 @ np.linalg.inv(R_00) @ R_01.T
            test_err = test_error(R_00, schur, R_01=R_01, alpha=closest_alpha, k=k, k_0=k_0)
            train_err = train_error(R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=closest_alpha, k=k, k_0=k_0)
            f_norm = np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T)
            
            # Initialize R_00 dict if it doesn't exist
            if R_00_str not in results:
                results[R_00_str] = {}
            
            # Store results
            results[R_00_str][alpha_str] = {
                "test_error": float(test_err),
                "train_error": float(train_err),
                "F_norm": float(f_norm),
                "diverged": False,
                "actual_alpha": float(closest_alpha)
            }
            print('storing', results[R_00_str][alpha_str])
            
            # Save after each iteration
            with open(filepath, 'w') as f:
                json.dump({
                    "metadata": {
                        "k": k,
                        "k_0": k_0,
                        "lambda_reg": lambda_reg
                    },
                    "results": results
                }, f, indent=2)
            
    return filepath

def read_fp_test_results(alpha, k, k_0, R_00, lambda_reg=0):
    """
    Read test results for the closest available alpha value.
    
    Args:
        alpha (float): The alpha value to look for
        k (int): Dimension of the system
        k_0 (int): Dimension of g_0
        R_00 (ndarray): Initial covariance matrix
        lambda_reg (float): Regularization parameter
    
    Returns:
        tuple: (test_error, train_error, F_norm, R_01, diverged) if found, None if not found
               where R_01 is the cross-correlation matrix (always zeros in current implementation)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_tests")
    
    if not os.path.exists(data_dir):
        print("No data directory found")
        return None
    
    filename = f"fp_test_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"No file found matching parameters k={k}, k_0={k_0}, lambda={lambda_reg}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        print(f"No results found for R_00={R_00}")
        return None
    
    available_alphas = [float(a) for a in data["results"][R_00_str].keys()]
    
    if not available_alphas:
        print(f"No valid results found for R_00={R_00}")
        return None
    
    available_alphas = np.array(available_alphas)
    closest_alpha = available_alphas[np.argmin(np.abs(available_alphas - alpha))]
    closest_alpha_str = str(closest_alpha)
    
    result = data["results"][R_00_str][closest_alpha_str]
    
    if result["diverged"]:
        return None, None, None, np.zeros((k, k_0)), True
    
    print(f"Found results in file: {filename}")
    print(f"Using alpha={closest_alpha} (requested alpha={alpha})")
    return (result["test_error"], result["train_error"], 
            result["F_norm"], np.zeros((k, k_0)), result["diverged"])

