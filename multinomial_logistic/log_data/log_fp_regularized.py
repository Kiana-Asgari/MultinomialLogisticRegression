import os
import json
import numpy as np
from state_evolution.full_recursion import state_evolution_full_recursion

# log the fp solution for alpha = [1.5, 2, 3, 5]
# for lambda in (0, 1.5)
# and R = I_k, and R_00 = [[1,1/2][1/2,1]
# and k = 2





def run_and_log_fp_regularized(k_0=2, k=2, alpha_values=[1.5,2,3,5], tol=1e-5, max_iter=200):     
    # Create base filename
    base_filename = f"fp_reg_data_k{k}_k0{k_0}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "fp_solution", base_filename)
    
    # Create data/fp_solution directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Load existing results if file exists
    if os.path.exists(base_filepath):
        print(f"Loading existing file: {base_filename}")
        with open(base_filepath, 'r') as f:
            existing_data = json.load(f)
            results = existing_data["results"]
    else:
        print(f"Creating new file: {base_filename}")
        results = {}

    # Specific values as per requirements
    R_00_values = np.array([np.eye(k), [[1,0.5], [0.5,1]]])
    alphas = np.array(alpha_values)  # Specific alpha values
    lambda_regs = np.concatenate([np.linspace(0.001,0.2,50), np.linspace(0.2, 1.5, 50)]).flatten()  # 100 points between 0 and 1.5
    lambda_regs = np.sort(lambda_regs)[::-1]

    for R_00 in R_00_values:
        for lambda_reg in lambda_regs:
            schur = R_00
            R_01 = np.zeros((k, k_0))
            S = np.eye(k)
            diverged_flag = False
            
            for alpha in alphas:  # Using specific alpha values
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
                
                if diverged_flag:
                    # If already diverged for smaller alpha, just log divergence
                    results[key] = {
                        "schur": np.zeros((k, k)).tolist(),
                        "R_01": np.zeros((k, k_0)).tolist(),
                        "S": np.zeros((k, k)).tolist(),
                        "diverged": True
                    }
                    
                    # Save after logging divergence
                    with open(base_filepath, 'w') as f:
                        json.dump({
                            "metadata": {
                                "k": k,
                                "k_0": k_0
                            },
                            "results": results
                        }, f, indent=2)
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
                
                results[key] = {
                    "schur": schur.tolist(),
                    "R_01": R_01.tolist(),
                    "S": S.tolist(),
                    "diverged": bool(diverged)
                }
                
                if diverged:
                    diverged_flag = True

                # Save after each computation
                with open(base_filepath, 'w') as f:
                    json.dump({
                        "metadata": {
                            "k": k,
                            "k_0": k_0
                        },
                        "results": results
                    }, f, indent=2)
            
    return base_filepath 