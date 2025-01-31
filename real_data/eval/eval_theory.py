import numpy as np
from real_data.eval.fit_data import fit_data
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.ESD.Marchenko_Pastur_FP import stieltjes_inversion
import os
import json
from scipy.linalg import sqrtm

from multinomial_logistic.evaluation.log_loss_test_error import test_error, irreducible_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from multinomial_logistic.evaluation.misclassification_test_error import misclassification_test_error, irreducible_misclassification_error





def eval_irreducible_error_theory(X_train, y_train, X_test, y_test,
                                  feature_name, n_hidden, file_number=1, seed=42):
    np.random.seed(seed)
    results = fit_data(X_train, y_train, X_test=X_test, y_test=y_test, compute_esd=False, seed=seed)
    Theta_hat = results['Theta_hat']
    R_00 = Theta_hat @ Theta_hat.T
    R_00_str = str(R_00.tolist())
    k=2
    k_0=2

    # Create base filename
    base_filename = f"irreducible_error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "error_theoretical", base_filename)
    print('base_filepath', base_filepath)
    
    # Create data directory if it doesn't exist
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
        
    # Skip if R_00 already exists
    if R_00_str in results:
        print(f"Skipping R_00 (already exists)")
        return base_filepath

    # Calculate irreducible errors
    irr_test_error = irreducible_error(R_00=R_00, k=k, k_0=k_0, alpha=0) #alpha does not matter 
    irr_misclass_error = irreducible_misclassification_error(R_00=R_00, k=k, k_0=k_0, alpha=0) #alpha does not matter 
    
    # Store results
    results[R_00_str] = {
        'irreducible_test_error': float(irr_test_error),
        'irreducible_misclassification_error': float(irr_misclass_error),
        'R_00': R_00.tolist()
    }

    # Save results
    with open(base_filepath, 'w') as f:
        json.dump({
            "metadata": {
                "k": k,
                "k_0": k_0,
                "feature name": feature_name,
                "n_hidden": n_hidden,
                "R_00": R_00_str
            },
            "results": results
        }, f, indent=2)
    
    return base_filepath







def eval_error_theory(X_train, y_train, X_test, y_test,
                    feature_name, n_hidden, file_number=1, seed=42):
    
    results = fit_data(X_train, y_train, X_test=X_test, y_test=y_test, compute_esd=False, seed=seed)
    Theta_hat = results['Theta_hat']
    R_00 = Theta_hat @ Theta_hat.T
    R_00_str = str(R_00.tolist())
    k=2
    k_0=2

   # Create base filename
    base_filename = f"error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "error_theoretical", base_filename)
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
        results[R_00_str] = {}

    ##############################################################################
    ##############################################################################  
    ##############################################################################
    alpha_values = np.linspace(5.44, 4.5, 10)
    for alpha in alpha_values:
        # Convert alpha to string for dictionary lookup
        alpha_str = str(alpha)
        
        # Skip if this alpha already exists
        if alpha_str in results[R_00_str]:
            print(f"Skipping alpha={alpha} (already exists)")
            continue
            
        schur, R_01, S, diverged = state_evolution_full_recursion(R_00=R_00,
                                                                 schur_0=R_00,
                                                                 R_01_0=np.zeros((2,2)),
                                                                 lambda_reg=0,
                                                                 alpha=alpha, 
                                                                 k=k, 
                                                                 k_0=k_0,
                                                                 max_iter=100,
                                                                 seed=seed,
                                                                 tol=1e-5)
        print(' found schur,', schur, 'R_01,', R_01, 'S,', S)

        A = R_01 @ np.linalg.inv(sqrtm(R_00))
        # Calculate test error, train error, and F_norm
        R_11 = schur + R_01 @ np.linalg.inv(R_00) @ R_01.T
        test_err = test_error(R_00, schur, R_01=R_01, alpha=alpha, k=k, k_0=k_0)
        train_err = train_error(R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=alpha, k=k, k_0=k_0)
        misclassification_test_err = misclassification_test_error(S=S, R_00=R_00,
                                                                           schur_t=schur, 
                                                                           A_t=A, 
                                                                           alpha=alpha, k=k, k_0=k_0)
        f_norm = np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T)
        results[R_00_str][alpha_str] = {
            'test_error': float(test_err),
            'train_error': float(train_err),
            'misclassification_test_error': float(misclassification_test_err),
            'f_norm': float(f_norm),
            'schur': schur.tolist(),
            'R_01': R_01.tolist(),
            'R_00': R_00.tolist(),
            'S': S.tolist()
        }
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
    
    return base_filepath

def eval_esd_hessian_theory(X_train, y_train, X_test, y_test, alpha, z_real_values,
                            feature_name, n_hidden, file_number=1, seed=42):

    results = fit_data(X_train, y_train, X_test=X_test, y_test=y_test, compute_esd=False, seed=seed)
    Theta_hat = results['Theta_hat']
    R_00 = Theta_hat @ Theta_hat.T
    R_00_str = str(R_00.tolist())
    k=2
    k_0=2

   # Create base filename
    base_filename = f"esd_data_alpha{alpha}_{file_number}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "data", "esd_theoretical", base_filename)
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
        results[R_00_str] = {str(alpha): {}}

    density_list = []
    last_MP_S = np.complex128(np.eye(k)) #initialize last_MP_S
    ##############################################################################
    ##############################################################################  
    ##############################################################################
    schur, R_01, S, diverged = state_evolution_full_recursion(R_00=R_00,
                                                                 schur_0=R_00,
                                                                 R_01_0=np.zeros((2,2)),
                                                                 lambda_reg=0,
                                                                 alpha=alpha, 
                                                                 k=k, 
                                                                 k_0=k_0,
                                                                 max_iter=100,
                                                                 seed=seed,
                                                                 tol=1e-5)
    print(' found schur,', schur, 'R_01,', R_01, 'S,', S)
    print('*****processing z_real min = ', z_real_values[0], ' max = ', z_real_values[-1], ' len = ', len(z_real_values))

    A = R_01 @ np.linalg.inv(sqrtm(R_00))

    for z_real in z_real_values:
        # choose z_imag based on z_real
        if z_real < 0.11:
            z_imag = 1e-4
        else:
            z_imag = 1e-3
        z_real_str = str(z_real)
        z_imag_str = str(z_imag)
        alpha_str = str(alpha)
        
        # Ensure the nested structure exists
        if R_00_str not in results:
            results[R_00_str] = {}
        if alpha_str not in results[R_00_str]:
            results[R_00_str][alpha_str] = {}
            
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
