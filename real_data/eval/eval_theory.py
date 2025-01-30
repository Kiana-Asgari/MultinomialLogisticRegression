import numpy as np
from real_data.eval.fit_data import fit_data
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.ESD.Marchenko_Pastur_FP import stieltjes_inversion
import os
import json
from scipy.linalg import sqrtm



def eval_esd_hessian_theory(X_train, y_train, X_test, y_test, alpha, z_real_values,
                            feature_name, n_hidden, file_number=1, seed=42):

    results = fit_data(X_train, y_train, X_test=X_test, y_test=y_test, compute_esd=False, seed=seed)
    Theta_hat = results['Theta_hat']
    R_00 = Theta_hat @ Theta_hat.T
    R_00_str = str(R_00.tolist())
    k=2
    k_0=2
    schur, R_01, S, diverged = state_evolution_full_recursion(R_00=R_00,
                                                                 schur_0=R_00,
                                                                 R_01_0=np.zeros((2,2)),
                                                                 lambda_reg=0,
                                                                 alpha=alpha, 
                                                                 k=k, 
                                                                 k_0=k_0,
                                                                 max_iter=100,
                                                                 seed=seed,
                                                                 tol=1e-1)
    print(' found schur,', schur, 'R_01,', R_01, 'S,', S)


    A = R_01 @ np.linalg.inv(sqrtm(R_00))

   # Create base filename
    base_filename = f"esd_data_alpha{alpha}_{file_number}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "esd_theoretical", base_filename)
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

    density_list = []

    ##############################################################################
    ##############################################################################  
    ##############################################################################
    for z_real in z_real_values:
        # choose z_imag based on z_real
        if z_real < 0.1:
            z_imag = 1e-4
        else:
            z_imag = 1e-3
        z_real_str = str(z_real)
        z_imag_str = str(z_imag)
        alpha_str = str(alpha)
        
        # More robust check for existing combinations
        if (z_real_str in results[R_00_str][alpha_str] and 
            results[R_00_str][alpha_str].get(z_real_str, {}).get(z_imag_str) is not None):
            print(f"Skipping z_real={z_real}, z_imag={z_imag} (already exists)")
            continue
            

        new_MP_S, density = stieltjes_inversion(R_00, schur, A, S, z_real=z_real, z_imag=z_imag,\
                                alpha=alpha, k=k, k_0=k_0, last_MP_S=last_MP_S, max_iter=500)
        print('at z_real = ', z_real, ' density = ', density)
        # Initialize z_real dict if needed
        if z_real_str not in results[R_00_str][alpha_str]:
            results[R_00_str][alpha_str][z_real_str] = {}
        
        # Store only density and new_MP_S values
        results[R_00_str][alpha_str][z_real_str][z_imag_str] = {
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
                    "alpha": alpha,
                    "feature name": feature_name,
                    "n_hidden": n_hidden,
                    "R_00": R_00_str
                },
                "results": results
            }, f, indent=2)
        print('Saved alpha = ', alpha, ' z_real = ', z_real, ' and z_imag = ', z_imag, ' density = ', density)
        density_list.append(density)
            
        print('final density list', density_list)
        print('final z_real values', z_real_values)
        print(' for alpha = ', alpha, ' and R_00 = ', R_00)
        return base_filepath