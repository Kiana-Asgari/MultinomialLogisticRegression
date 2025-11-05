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
from typing import Literal
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline
import warnings
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import fit_mle_skitlearn
from multinomial_logistic.MLE_empirical.ESD_empirical import esd_empirical
from configs.R_initiation import get_R_00



def log_mle_esd(k, k_0, d=250, n_trials=20):
   
    R_00 = np.array([[1,1/2], [1/2,1]])
    
    # Create the directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "newdata", "mle_empirical", "esd")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create filename
    filename = f"esd_k{k}_k0{k_0}_d{d}_ntrials{n_trials}.json"
    filepath = os.path.join(data_dir, filename)
    
    # Compute ESDs
    esd_full_3 = read_mle_esd(k, k_0, alpha=3.0)
    print('esd_full_3 shape:', esd_full_3.shape)
    esd_full_5 = read_mle_esd(k, k_0, alpha=5.0)
    print('esd_full_5 shape:', esd_full_5.shape)
    esd_full_10 = read_mle_esd(k, k_0, alpha=10.0)
    print('esd_full_10 shape:', esd_full_10.shape)
    esd_full_20 = esd_empirical(alpha=20, k=k, lambda_reg=0, R_00=R_00, n_trials=n_trials, d=d, skitlearn=True)
    print('esd_full_20 shape:', esd_full_20.shape)
    
    # Prepare data for saving
    data = {
        "metadata": {
            "k": k,
            "k_0": k_0,
            "d": d,
            "n_trials": n_trials,
            "R_00": R_00.tolist()
        },
        "results": {
            "3": esd_full_3.tolist() if esd_full_3 is not None else None,
            "5": esd_full_5.tolist() if esd_full_5 is not None else None,
            "10": esd_full_10.tolist() if esd_full_10 is not None else None,
            "20": esd_full_20.tolist() if esd_full_20 is not None else None
        }
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"ESD data saved to: {filepath}")
    return filepath

def run_and_log_mle_regularized(k_0, k, lambda_reg=0, d=250, n_trials=100):
     # Create base filename without timestamp, but with d and n_trials
    base_filename = f"mle_reg_k{k}_k0{k_0}_d{d}_ntrials{n_trials}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "newdata", "mle_empirical", base_filename)
    
    # Create data/mle_empirical directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Check for existing files with matching parameters
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        if f"mle_reg_k{k}_k0{k_0}_d{d}_ntrials{n_trials}" in filename:
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
        print(f"Creating new  mle reg file: {os.path.basename(filepath)}")

    R_00_values = np.array([ [[1,0.5], [0.5,1]]])  
    alpha_values=[1.5,3,5,10]
    #alpha_values = [3]
    #lambda_regs = [0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1] # 100 points between 0 and 1.5
    lambda_regs = np.linspace(0.01, 0.6, 30)
    n_trials = 100
    d = 250

    for R_00 in R_00_values:
        for lambda_reg in lambda_regs:

            for alpha in alpha_values:  # Using specific alpha values
                # Create composite key
                key = json.dumps({
                    "R_00": R_00.tolist(),
                    "alpha": float(alpha),
                    "lambda_reg": float(lambda_reg)
                })
                
                # Skip if we already have results for this combination
                if key in results:
                    print(f"Skipping alpha={alpha}, lambda={lambda_reg} for R_00={R_00} (already exists)")
                    continue
                    
                print(f"\nProcessing alpha={alpha}, lambda={lambda_reg}")
                
                
                Theta_hats, norms, test_errors, train_errors, misclassification_test_errors = fit_mle_baseline(
                        alpha=alpha,
                        k=k,
                        lambda_reg=lambda_reg*2, # remember the *2
                        R_00=R_00,
                        n_trials=n_trials,
                        d=d,
                        return_full_results=True
                    )
                
                results[key] = {
                    "test_errors": test_errors.tolist(),
                    "train_errors": train_errors.tolist(),
                    "norms": norms.tolist(),
                    "misclassification_test_errors": misclassification_test_errors.tolist()
                }
                
                print('done mle for alpha=', alpha, 'lambda=', lambda_reg, 'avg test error=', np.mean(test_errors), 'avg train error=', np.mean(train_errors), 'avg norm=', np.mean(norms), 'avg misclassification test error=', np.mean(misclassification_test_errors))
                # Save after each computation
                with open(base_filepath, 'w') as f:
                    json.dump({
                        "metadata": {
                            "k": k,
                            "k_0": k_0
                        },
                        "results": results
                    }, f, indent=2)
            
            
    return filepath

def run_and_log_mle(k_0, k, lambda_reg=0, d=250,
                    n_trials=100, 
                    two_classes_close=False,
                    non_symmetric=False,
                    type_3:Literal[False, 'symmetric', 'two_classes_close', 'three_classes_close', 'two_vs_two_vs_one'] = False):     
    # Set parameters
    
    # Create base filename without timestamp, but with d and n_trials
    if type_3==False and non_symmetric:
        base_filename = f"mle_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}_non_symmetric.json"
    elif type_3==False and two_classes_close:
        base_filename = f"mle_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}_two_classes_close.json"
    elif type_3==False and not non_symmetric and not two_classes_close:
        base_filename = f"mle_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}_k3.json"
    elif type_3 != False:
        base_filename = f"MLE_evals(d={d},ntrials={n_trials})_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"

    if type_3 != False:
        base_filepath = os.path.join(os.path.dirname(__file__), "Oct_data", "mle_empirical", base_filename)
    else:
        base_filepath = os.path.join(os.path.dirname(__file__), "newdata", "mle_empirical", base_filename)
    
    # Create data/mle_empirical directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Check for existing files with matching parameters
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        if base_filename == filename:
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

    R_00_values = np.array([[[1,0.5], [0.5,1]]])  

    if type_3==False and two_classes_close:
        R_00_values = np.array([[[1,0.9], [0.9,1]]])
    elif type_3==False and non_symmetric:
        R_00_values = np.array([[[1,-0.5], [-0.5,1]]])
    elif type_3==False and not non_symmetric and not two_classes_close:
        R_00_values = np.array([np.eye(3)])
    elif type_3 != False:
        R_00_values = np.array([get_R_00(k, type_3)])


    alphas = np.arange(3.6, 16, 0.4) 
    alphas = alphas[::-1]
    print('\n MEL:alphas:', alphas.shape, 'R_00_values:', R_00_values, 'path:', base_filepath, '\n')

    for R_00 in R_00_values:
        for alpha in alphas:
            alpha_str = str(alpha)
            R_00_str = str(R_00.tolist())
            
            # Skip if we already have results for this alpha and R_00
            if R_00_str in results and alpha_str in results[R_00_str]:
                print(f"Skipping alpha={alpha} for R_00={R_00} (already exists)")
                continue
                
            print(f"\nMLE: Running alpha = {alpha}")
            
            # Set up warning catching
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    emp_results = fit_mle_skitlearn(
                        alpha=alpha,
                        k=k,
                        R_00=R_00,
                        n_trials=n_trials,
                        d=d
                    )
                    norms, test_errors, train_errors = emp_results['norms'], emp_results['test_errors'], emp_results['train_errors']
                    misclassification_test_errors = emp_results['misclass_test_errors']
                    print('for alpha=', alpha, 'misclassification_test_errors=', np.mean(misclassification_test_errors), np.std(misclassification_test_errors))
                    # Check if overflow warning was raised
                    if any(issubclass(warn.category, RuntimeWarning) and "overflow" in str(warn.message) for warn in w):
                        raise FloatingPointError("Overflow detected in computation")
                    
                    # Check if average norm exceeds threshold
                    avg_norm = np.mean(norms)
                    diverged = False
                    
                    # Initialize R_00 dict if it doesn't exist
                    if R_00_str not in results:
                        results[R_00_str] = {}
                    
                    if diverged:
                        # If diverged due to large norm, save empty arrays
                        results[R_00_str][alpha_str] = {
                            "norm": [],
                            "test_errors": [],
                            "train_errors": [],
                            "misclassification_test_errors": [],
                            "shapes": {
                                "norms": [0],
                                "test_errors": [0],
                                "train_errors": [0],
                                "misclassification_test_errors": [0]
                            },
                            "diverged": True
                        }
                        print(f"Diverged due to large norm (avg_norm={avg_norm:.2f}) for alpha={alpha}")
                    else:
                        results[R_00_str][alpha_str] = {
                            "norm": norms.tolist(),
                            "test_errors": test_errors.tolist(),
                            "train_errors": train_errors.tolist(),
                            "misclassification_test_errors": misclassification_test_errors.tolist(),
                            "shapes": {
                                "norms": list(norms.shape),
                                "test_errors": list(test_errors.shape),
                                "train_errors": list(train_errors.shape),
                                "misclassification_test_errors": list(misclassification_test_errors.shape)
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
                        "misclassification_test_errors": [],
                        "shapes": {
                            "norms": [0],
                            "test_errors": [0],
                            "train_errors": [0],
                            "misclassification_test_errors": [0]
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




def run_and_log_esd_empirical(k_0, k, alpha, R_00, lambda_reg=0, n_iters=100, d=250):
    # Create base filename
    base_filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_alpha{alpha:.2f}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "mle_empirical", base_filename)
    
    # Create data/mle_empirical directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "metadata": {
            "k": k,
            "k_0": k_0,
            "lambda_reg": lambda_reg,
            "d": d,
            "alpha": alpha,
            "n_iters": n_iters,
            "R_00": R_00.tolist()
        },
        "eigenvalues": []
    }
    
    # Run iterations
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
    
        print(f"\rComputing ESD")
        
        try:
            # Compute ESD for this iteration
            avg_Theta_hat, avg_esd, eigenvalues = esd_empirical(
                alpha=alpha,
                k=k,
                lambda_reg=lambda_reg,
                R_00=R_00,
                d=d,
                max_iter=n_iters
            )
            
            # Check for valid eigenvalues
            if eigenvalues is not None:
                eigenvalues = np.asarray(eigenvalues)
                if not np.any(np.isnan(eigenvalues.flatten())):  # Check for NaN in flattened array
                    results["eigenvalues"].append(eigenvalues.tolist())
            
        except (FloatingPointError, RuntimeError) as e:
            print(f"\nNumerical error in iteration {iter}: {str(e)}")

    
    print("\nSaving results...")
    
    # Save results
    with open(base_filepath, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to: {base_filepath}")
    return base_filepath



