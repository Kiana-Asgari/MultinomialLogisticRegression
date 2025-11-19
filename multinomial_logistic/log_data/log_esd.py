import numpy as np
from scipy.linalg import sqrtm
from typing import Literal
import json
import os
from multinomial_logistic.log_data.log_fp import read_fp_results
from multinomial_logistic.ESD.Marchenko_Pastur_FP_GPU import stieltjes_inversion
from configs.R_initiation import get_R_00
from state_evolution.full_recursion import state_evolution_full_recursion

def run_and_log_esd(k_0, k, lambda_reg=0, alpha_input=None, R_00_input=None, 
                    S_input=None, schur_input=None, R_01_input=None, file_number=None, z_imag=1e-4,
                    z_real_values = np.linspace(0.5, 1, 10),
                    tol=1e-4, max_iter=200, type_3:Literal[False, 'symmetric', 'two_classes_close', 'three_classes_close'] = False):     
    # Create base filename
    if file_number is None:
        base_filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}_alpha{alpha_input}.json"
    else:
        base_filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}_file{file_number}.json"
    # if k<3:
    #     base_filepath = os.path.join(os.path.dirname(__file__), "newdata", "esd", base_filename)
    # else:
    base_filepath = os.path.join(os.path.dirname(__file__), "Oct_data", "esd", base_filename)
    print('base_filepath', base_filepath)
    
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


            
    R_00_values = [R_00_input]
    alphas = [alpha_input]

    alphas = sorted(list(alphas), reverse=True)  # Sort in decreasing order


    for R_00 in R_00_values:
        R_00_str = str(R_00.tolist())
        if R_00_str not in results:
            results[R_00_str] = {}
            
        for alpha in alphas:
            density_list = []
            print(f"\nProcessing alpha = {alpha}")
        

            # Get FP solution results from file
            fp_results = read_fp_results(alpha, k, k_0, R_00, lambda_reg, type_3=type_3)

            if fp_results is None:
                print(f"No FP solution found for alpha={alpha}, running state evolution")
                continue
                schur, R_01, S, diverged = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k, k)), S_0=np.eye(k),
                                                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0,
                                                                         max_iter=max_iter)
                
                    
                actual_alpha = alpha
                schur, R_01, S = np.array(schur.cpu()), np.array(R_01.cpu()), np.array(S.cpu())
                
            else:
                schur, R_01, S, diverged, actual_alpha = fp_results
     

            A = R_01 @ np.linalg.inv(sqrtm(R_00))
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
                z_real_str = str(z_real)
                z_imag_str = str(z_imag)
                
                # More robust check for existing combinations
                if (z_real_str in results[R_00_str][actual_alpha_str] and 
                    results[R_00_str][actual_alpha_str].get(z_real_str, {}).get(z_imag_str) is not None):
                    print(f"Skipping z_real={z_real}, z_imag={z_imag} (already exists)")
                    continue
                    

                new_MP_S, density = stieltjes_inversion(R_00, schur, A, S, z_real=z_real, z_imag=z_imag,\
                                       alpha=alpha, k=k, k_0=k_0, last_MP_S=last_MP_S,
                                        max_iter=max_iter, tol=tol)
                print('at z_real = ', z_real, ' density = ', density)
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
                print('Saved alpha = ', alpha, ' z_real = ', z_real, ' and z_imag = ', z_imag, ' density = ', density)
                density_list.append(density)
    print('final density list', density_list)
    print('final z_real values', z_real_values)
    print(' for alpha = ', alpha, ' and R_00 = ', R_00)
    return base_filepath

def read_esd_results(alpha, k, k_0, R_00, z_real, z_imag=1e-4, lambda_reg=0):

    data_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "esd")
    
    if not os.path.exists(data_dir):
        print("No data directory found")
        return None, None, None
    
    filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}_alpha{alpha}.json"
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





def get_density_data(k, k_0, R_00, alpha_target, lambda_reg=0, z_imag_target=1e-4,
                      clean_data_for_5=False, clean_data_for_3=True):
    # Load the ESD data file
    data_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "esd")
    filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}_alpha{alpha_target}.json"
    filepath = os.path.join(data_dir, filename)

    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        raise ValueError(f"R_00={R_00} not found in data")
        

    
    # Find exact alpha match
    alpha_str_int = str(int(alpha_target)) if int(alpha_target) == alpha_target else str(alpha_target)
    alpha_target_str = alpha_str_int + '.0'

    if alpha_target_str not in data["results"][R_00_str] and alpha_str_int not in data["results"][R_00_str]:
        print(f"Alpha {alpha_target} was not found in the data")
        available_alphas = sorted([float(a) for a in data["results"][R_00_str].keys()])
        print(f"Available alphas: {available_alphas}")
        return None, None, None, None
    if alpha_str_int in data["results"][R_00_str]:
        alpha_target_str = alpha_str_int

    
    alpha_data = data["results"][R_00_str][alpha_target_str]
    
    if alpha_data.get("diverged", False):
        print(f"Solution diverged for alpha={alpha_target}")
        return None, None, None, None
    
    # Get all z_real values and sort them
    z_reals = []
    z_imags = []
    densities = []
    
    # Convert z_real strings to floats for proper sorting
    z_real_values = sorted([float(z) for z in alpha_data.keys() if z != "diverged"])
    
    for z_real in z_real_values:
        z_real_str = str(z_real)
        if z_real_str not in alpha_data:
            continue
            
        z_imag = min(float(z_imag) for z_imag in alpha_data[z_real_str].keys())
        
        z_imag_str = str(z_imag)
        density = alpha_data[z_real_str][z_imag_str]['density']
        
        z_reals.append(z_real)
        z_imags.append(z_imag_target)
        densities.append(density)

    
    return alpha_target, np.array(z_reals), np.array(z_imags), np.array(densities)








