import numpy as np
import json
import os
from datetime import datetime
from state_evolution.full_recursion import state_evolution_full_recursion
import fcntl
from typing import Literal
from configs.R_initiation import get_R_00

def run_and_log_fp(k_0, k, lambda_reg=0, tol=1e-5, max_iter=300, non_symmetric=False, two_classes_close=False, 
                    type_3:Literal[False, 'symmetric', 'two_classes_close', 'three_classes_close'] = False):     
    # Create base filename without timestamp

    if non_symmetric and not type_3:
        R_00_values = np.array([[[1,-0.5], [-0.5,1]]])
    elif two_classes_close and not type_3:
        R_00_values = np.array([[[1,0.9], [0.9,1]]])
    elif type_3 != False:
        R_00_values = np.array([get_R_00(k, type_3)])
    else:
        raise ValueError("Invalid combination of parameters")

    R_00_str = str(R_00_values[0].tolist())

    alphas = np.arange(5, 20, 0.5)
    alphas = alphas[::-1]
    print('running alphas:', alphas)


    if not non_symmetric and not two_classes_close and not type_3:
        print("Running symmetric FP")
        base_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
        base_filename_dir = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}"
        base_filepath = os.path.join(os.path.dirname(__file__), "tempdata", "fp_solution", base_filename)
    elif not type_3 and non_symmetric:
        print("Running non-symmetric FP")
        base_filename =  f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        base_filename_dir = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric"
        base_filepath = os.path.join(os.path.dirname(__file__), "tempdata", "fp_solution_nonsym", base_filename)
    elif not type_3 and two_classes_close:
        print("Running two classes close FP")
        base_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        base_filename_dir = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close"
        base_filepath = os.path.join(os.path.dirname(__file__), "tempdata", "fp_solution_two_classes_close", base_filename)
    elif type_3 != False:
        print("Running 3 classes FP for type_3 =", type_3)
        base_filename = f"FP_solutions_(k={k},k0={k_0},lambda={lambda_reg})_{type_3}.json"
        base_filepath = os.path.join(os.path.dirname(__file__), "Oct_data", "fp_solution", base_filename)

    
    # Create data/fp_solution directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    print('saving fp solutions to base_filepath:', base_filepath)
    print('\nR_00 =', R_00_values)

    
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
        results = existing_data["results"].copy()  # Create a copy of existing results
        alphas_existing = np.array([float(alpha) for alpha in results[R_00_str]])
        print(f"Appending to existing file: {filename}, alphas = {alphas_existing}")
    else:
        # Create new file
        filepath = base_filepath
        results = {}
        print(f"Creating new file: {os.path.basename(filepath)}")





    for R_00 in R_00_values:
        schur = R_00
        R_01 = np.zeros((k, k_0))
        S = np.eye(k)
        diverged_flag = False  # Track divergence for this R_00
    
        for alpha in alphas:  # Note: alphas are already sorted in decreasing order

            alpha_str = str(alpha)
            R_00_str = str(R_00.tolist())
            
            # Skip if we already have results for this alpha and R_00
            if R_00_str in results and alpha_str in results[R_00_str] and results[R_00_str][alpha_str]["diverged"] == False:
                print(f"Skipping alpha={alpha} for R_00={R_00} (already exists)")
                continue
            print(f"\nRunning alpha = {alpha}")

            if R_00_str in results and alpha_str in results[R_00_str]: #and results[R_00_str][alpha_str]["diverged"] == True:
                schur = np.array(results[R_00_str][alpha_str]["schur"])
                R_01 = np.array(results[R_00_str][alpha_str]["R_01"])
                S = np.array(results[R_00_str][alpha_str]["S"])
                print("Using previous results for alpha=", alpha," to continues the state evolution")


            schur, R_01, S, diverged = state_evolution_full_recursion(
                                                            R_00=R_00,
                                                            schur_0=schur,
                                                            R_01_0=R_01,
                                                            S_0=S,
                                                            lambda_reg=lambda_reg,
                                                            alpha=alpha,
                                                            k=k,
                                                            k_0=k_0,
                                                            tol=tol,
                                                            max_iter=max_iter
                                                        )
            

            # Initialize R_00 dict if it doesn't exist
            if R_00_str not in results:
                results[R_00_str] = {}
            
            # Always update with new results
            results[R_00_str][alpha_str] = {
                "schur": schur.tolist(),
                "R_01": R_01.tolist(),
                "S": S.tolist(),
                "diverged": bool(False)
            }
            


            # Save after each iteration
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump({
                        "metadata": {
                            "k": k,
                            "k_0": k_0,
                            "lambda_reg": lambda_reg
                        },
                        "results": results
                    }, f, indent=2)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            # Atomic rename
            os.replace(temp_filepath, filepath)
            
    return filepath

def read_fp_results(alpha, k, k_0, R_00, lambda_reg=0, non_symmetric=False, two_classes_close=False):
    """
    Read results for the closest available alpha value from the matching file.
    
    Args:
        alpha (float): The alpha value to look for
        k (int): Dimension of the system
        k_0 (int): Dimension of g_0
        R_00 (ndarray): Initial covariance matrix
        lambda_reg (float): Regularization parameter
    
    Returns:
        tuple: (schur, S, diverged) if found, None if not found
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_solution")
    
    if not os.path.exists(data_dir):
        print("No data directory found")
        return None
    
    # Look for the specific file
    if two_classes_close:
        fp_data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_solution_two_classes_close")
        filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_two_classes_close.json"
        filepath = os.path.join(fp_data_dir, filename)
    elif non_symmetric:
        fp_data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_solution_nonsym")
        filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}_non_symmetric.json"
        filepath = os.path.join(fp_data_dir, filename)
    else:
        fp_data_dir = os.path.join(os.path.dirname(__file__), "data", "fp_solution")
        filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
        filepath = os.path.join(fp_data_dir, filename)
    
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
    available_alphas = [float(a) for a in data["results"][R_00_str].keys() 
                       if "error" not in data["results"][R_00_str][a]]
    
    if not available_alphas:
        print(f"No valid results found for R_00={R_00}")
        return None
    
    # Find closest alpha
    available_alphas = np.array(available_alphas)
    closest_alpha = available_alphas[np.argmin(np.abs(available_alphas - alpha))]
    closest_alpha_str = str(closest_alpha)
    
    # Try both integer and float string formats for integer alphas
    if False:#closest_alpha.is_integer():
        
        int_alpha_str = str(int(closest_alpha))
        float_alpha_str = f"{closest_alpha}.0"
        if int_alpha_str in data["results"][R_00_str]:
            result = data["results"][R_00_str][int_alpha_str]
        else:
            result = data["results"][R_00_str][float_alpha_str]
    else:
        result = data["results"][R_00_str][closest_alpha_str]
    
    # Convert lists back to numpy arrays
    schur = np.array(result["schur"]).reshape(k,k)
    R_01 = np.array(result["R_01"]).reshape(k_0,k)
    S = np.array(result["S"]).reshape(k,k)
    diverged = result["diverged"]
    
    print(f"Found FP results in file: {filename}")
    print(f"Using alpha={closest_alpha} (requested alpha={alpha} not found)")
    print(f"R_00 = {R_00}, S = {S}, schur = {schur}")
    return schur, R_01, S, diverged, closest_alpha
