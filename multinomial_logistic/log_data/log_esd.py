import numpy as np
import json
import os
from multinomial_logistic.log_data.log_fp import read_fp_results
from multinomial_logistic.ESD_theoretical.Marchenko_Pastur_FP import stieltjes_inversion

def run_and_log_esd(k_0, k, lambda_reg=0, alpha_input=None, R_00_input=None, 
                    read_from_file=True, S_input=None, schur_input=None, R_01_input=None):     
    # Create base filename
    base_filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "esd", base_filename)
    
    # Create data/esd directory if it doesn't exist
    os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
    
    # Initialize or load existing results
    if os.path.exists(base_filepath):
        print(f"Loading existing file: {os.path.basename(base_filepath)}")
        with open(base_filepath, 'r') as f:
            existing_data = json.load(f)
            results = existing_data["results"]
    else:
        print(f"Creating new file: {os.path.basename(base_filepath)}")
        results = {}

    if read_from_file:
        # Original file reading logic for FP solutions
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

        if alpha_input is None:
            for R_00_str in fp_data["results"].keys():
                R_00 = np.array(json.loads(R_00_str))
                R_00_values.append(R_00)
            alphas.update(float(alpha) for alpha in fp_data["results"][R_00_str].keys())
        else:
            R_00_values.append(R_00_input)
            alphas.add(alpha_input)
    else:
        # Use input values directly
        if alpha_input is None or R_00_input is None or S_input is None or schur_input is None or R_01_input is None:
            raise ValueError("When read_from_file is False, all input parameters must be provided")
        R_00_values = [R_00_input]
        alphas = [alpha_input]

    alphas = sorted(list(alphas), reverse=True)  # Sort in decreasing order
    z_real_values = np.concatenate([np.linspace(0.0001, 0.08, 200), np.linspace(0.08, 0.8, 50)]).flatten()
    z_real_values = np.sort(z_real_values)

    for R_00 in R_00_values:
        R_00_str = str(R_00.tolist())
        if R_00_str not in results:
            results[R_00_str] = {}
            
        for alpha in alphas:
            print(f"\nProcessing alpha = {alpha}")
            
            if read_from_file:
                # Get FP solution results from file
                fp_results = read_fp_results(alpha, k, k_0, R_00, lambda_reg)
                if fp_results is None:
                    print(f"No FP solution found for alpha={alpha}")
                    continue
                schur, R_01, S, diverged, actual_alpha = fp_results
            else:
                # Use input values directly
                schur = schur_input
                R_01 = R_01_input
                S = S_input
                diverged = False
                actual_alpha = alpha

            A = R_01 @ np.linalg.inv(np.sqrt(R_00))
            actual_alpha_str = str(actual_alpha)
            
            if actual_alpha_str not in results[R_00_str]:
                results[R_00_str][actual_alpha_str] = {}
                
            if diverged:
                results[R_00_str][actual_alpha_str]["diverged"] = True
                continue

            results[R_00_str][actual_alpha_str]["diverged"] = False
            last_MP_S = np.complex128(np.eye(k))
            print('Starting to recover density for alpha = ', actual_alpha, ' and R_00 = ', R_00)

            ##############################################################################
            ##############################################################################  
            ##############################################################################
            for z_real in z_real_values:
                # choose z_imag based on z_real
                if z_real < 0.003:
                    z_imag = 1e-5
                elif z_real < 0.08:
                    z_imag = 1e-4
                else:
                    z_imag = 1e-3
                z_real_str = str(z_real)
                z_imag_str = str(z_imag)
                
                # More robust check for existing combinations
                if (z_real_str in results[R_00_str][actual_alpha_str] and 
                    results[R_00_str][actual_alpha_str].get(z_real_str, {}).get(z_imag_str) is not None):
                    print(f"Skipping z_real={z_real}, z_imag={z_imag} (already exists)")
                    continue

                new_MP_S, density = stieltjes_inversion(
                    R_00=R_00, 
                    schur=schur, 
                    A=A, 
                    S=S, 
                    z_real=z_real,
                    z_imag=z_imag,
                    alpha=alpha,
                    k=k,
                    k_0=k_0,
                    last_MP_S=last_MP_S
                )
                
                # Initialize z_real dict if needed
                if z_real_str not in results[R_00_str][actual_alpha_str]:
                    results[R_00_str][actual_alpha_str][z_real_str] = {}
                
                # Store only density and new_MP_S values
                results[R_00_str][actual_alpha_str][z_real_str][z_imag_str] = {
                    'density': float(density),
                    'MP_S': {
                        'real': new_MP_S.real.tolist(),
                        'imag': new_MP_S.imag.tolist()
                    }
                }
                last_MP_S = new_MP_S

                # Save after each computation
                with open(base_filepath, 'w') as f:
                    json.dump({
                        "metadata": {
                            "k": k,
                            "k_0": k_0,
                            "lambda_reg": lambda_reg
                        },
                        "results": results
                    }, f, indent=2)
            
    return base_filepath

def read_esd_results(alpha, k, k_0, R_00, z_real, z_imag=1e-4, lambda_reg=0):
    """
    Read ESD results for the closest available alpha and z_real values.
    
    Returns:
        tuple: (density, MP_S, diverged) if found, (None, None, None) if not found
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "esd")
    
    if not os.path.exists(data_dir):
        print("No data directory found")
        return None, None, None
    
    filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}.json"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"No file found matching parameters k={k}, k_0={k_0}, lambda={lambda_reg}")
        return None, None, None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        print(f"No results found for R_00={R_00}")
        return None, None, None
    
    # Find closest alpha
    available_alphas = [float(a) for a in data["results"][R_00_str].keys()]
    if not available_alphas:
        return None, None, None
    
    closest_alpha = available_alphas[np.argmin(np.abs(np.array(available_alphas) - alpha))]
    alpha_str = str(closest_alpha)
    
    if data["results"][R_00_str][alpha_str].get("diverged", False):
        return None, None, True
    
    # Find closest z_real
    available_z_reals = [float(z) for z in data["results"][R_00_str][alpha_str].keys() 
                        if z != "diverged"]
    if not available_z_reals:
        return None, None, None
    
    closest_z_real = available_z_reals[np.argmin(np.abs(np.array(available_z_reals) - z_real))]
    z_real_str = str(closest_z_real)
    
    # Find closest z_imag
    z_imag_str = str(z_imag)
    if z_imag_str not in data["results"][R_00_str][alpha_str][z_real_str]:
        print(f"No results found for z_imag={z_imag}")
        return None, None, None
    
    result = data["results"][R_00_str][alpha_str][z_real_str][z_imag_str]
    
    # Handle both old format (just density) and new format (dict with density and MP_S)
    if isinstance(result, dict):
        density = result['density']
        MP_S = np.array(result['MP_S']['real']) + 1j * np.array(result['MP_S']['imag'])
    else:
        # Legacy format: only density was stored
        density = result
        MP_S = None
    
    print(f"Found results in file: {filename}")
    print(f"Using alpha={closest_alpha}, z_real={closest_z_real}, z_imag={z_imag}")
    return density, MP_S, False