import numpy as np
import json
import os
from datetime import datetime
from state_evolution.full_recursion import state_evolution_full_recursion

def run_and_log_fp(k_0, k, lambda_reg=0, tol=1e-5, max_iter=200):     
    # Create base filename without timestamp
    base_filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "fp_solution", base_filename)
    
    # Create data/fp_solution directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Check for existing files with matching parameters
    data_dir = os.path.dirname(base_filepath)
    existing_files = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        if f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}" in filename:
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
    alphas = np.sort(alphas)[::-1]
  

    for R_00 in R_00_values:
        schur = R_00
        R_01 = np.zeros((k, k_0))
        S = np.eye(k)
        diverged_flag = False  # Track divergence for this R_00
    
        for alpha in alphas:  # Note: alphas are already sorted in decreasing order
            alpha_str = str(alpha)
            R_00_str = str(R_00.tolist())
            
            # Skip if we already have results for this alpha and R_00
            if R_00_str in results and alpha_str in results[R_00_str]:
                print(f"Skipping alpha={alpha} for R_00={R_00} (already exists)")
                continue
                
            print(f"\nRunning alpha = {alpha}")
            
            if diverged_flag:
                # If already diverged for smaller alpha, just log divergence
                results[R_00_str][alpha_str] = {
                    "schur": np.zeros((k, k)).tolist(),
                    "R_01": np.zeros((k, k_0)).tolist(),
                    "S": np.zeros((k, k)).tolist(),
                    "diverged": True
                }
                continue
            
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
            
            results[R_00_str][alpha_str] = {
                "schur": schur.tolist(),
                "R_01": R_01.tolist(),
                "S": S.tolist(),
                "diverged": bool(diverged)
            }
            
            if diverged:
                diverged_flag = True  # Mark as diverged for future alphas

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

def read_fp_results(alpha, k, k_0, R_00, lambda_reg=0):
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
    filename = f"fp_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
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
    available_alphas = [float(a) for a in data["results"][R_00_str].keys() 
                       if "error" not in data["results"][R_00_str][a]]
    
    if not available_alphas:
        print(f"No valid results found for R_00={R_00}")
        return None
    
    # Find closest alpha
    available_alphas = np.array(available_alphas)
    closest_alpha = available_alphas[np.argmin(np.abs(available_alphas - alpha))]
    closest_alpha_str = str(closest_alpha)
    
    result = data["results"][R_00_str][closest_alpha_str]
    
    # Convert lists back to numpy arrays
    schur = np.array(result["schur"]).reshape(k,k)
    R_01 = np.array(result["R_01"]).reshape(k_0,k)
    S = np.array(result["S"]).reshape(k,k)
    diverged = result["diverged"]
    
    print(f"Found results in file: {filename}")
    print(f"Using alpha={closest_alpha} (requested alpha={alpha} not found)")
    return schur, R_01, S, diverged, closest_alpha