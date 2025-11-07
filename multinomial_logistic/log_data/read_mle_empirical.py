import os
import json
import numpy as np


def read_mle_results(alpha, k, k_0, R_00, lambda_reg=0, d=250, n_trials=100, compressed=True):
    # Determine the appropriate directory based on compressed flag
    base_dir = "mle_empirical_compressed" if compressed else "mle_empirical"
    data_dir = os.path.join(os.path.dirname(__file__), "data", base_dir)
    
    
    # Look for the specific file with d and n_trials in the name
    filename = f"mle_data{'_compressed' if compressed else ''}_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}.json"
    filepath = os.path.join(data_dir, filename)
    
    # Read the file
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Check if we have results for this R_00
    R_00_str = str(R_00.tolist())
    
    # Get all available alphas for this R_00
    available_alphas = [float(a) for a in data["results"][R_00_str].keys()]
    
    
    # Find closest alpha
    available_alphas = np.array(available_alphas)
    closest_alpha = available_alphas[np.argmin(np.abs(available_alphas - alpha))]
    closest_alpha_str = str(closest_alpha)
    
    result = data["results"][R_00_str][closest_alpha_str]
    
    # Convert lists back to numpy arrays with proper shapes
    shapes = result["shapes"]
    norm = np.array(result["norm"]).reshape(shapes["norms"])
    test_errors = np.array(result["test_errors"]).reshape(shapes["test_errors"])
    train_errors = np.array(result["train_errors"]).reshape(shapes["train_errors"])
    
    # Handle theta_hats based on compressed flag
    if compressed:
        theta_hats = None
    else:
        theta_hats = np.array(result["theta_hats"]).reshape(shapes["theta_hats"])
    
    print(f"Found results in file: {filename}")
    print(f"Using alpha={closest_alpha} (requested alpha={alpha})")
    return norm, test_errors, train_errors, theta_hats, False




def get_mle_statistics(k, k_0, R_00, lambda_reg=0, d=250, n_trials=100):
    """
    Read compressed MLE data and compute statistics for each alpha value.
    
    Args:
        k (int): Dimension of the system
        k_0 (int): Dimension of g_0
        R_00 (ndarray): Initial covariance matrix
        lambda_reg (float): Regularization parameter
        d (int): Dimension of the problem
        n_trials (int): Number of trials
    
    Returns:
        tuple: (alphas, norm_means, norm_stds, test_means, test_stds, train_means, train_stds)
        where:
        - alphas: list of alpha values
        - norm_means: list of mean norms for each alpha
        - norm_stds: list of norm standard deviations
        - test_means: list of mean test errors
        - test_stds: list of test error standard deviations
        - train_means: list of mean train errors
        - train_stds: list of train error standard deviations
        Returns None if data not found
    """
    # Get the filepath
    filename = f"mle_data_compressed_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_ntrials{n_trials}.json"
    filepath = os.path.join(os.path.dirname(__file__), "data", "mle_empirical_compressed", filename)
    
    if not os.path.exists(filepath):
        print(f"No compressed file found: {filename}")
        return None
    
    # Read the file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if we have results for this R_00
    R_00_str = str(R_00.tolist())
    if R_00_str not in data["results"]:
        print(f"No results found for R_00={R_00}")
        return None
    
    # Initialize lists to store results
    alphas = []
    norm_means = []
    norm_stds = []
    test_means = []
    test_stds = []
    train_means = []
    train_stds = []
    
    # Process each alpha value
    for alpha_str, alpha_data in data["results"][R_00_str].items():
        # Skip if diverged
        if alpha_data["diverged"]:
            continue
        
        # Convert strings to float/arrays
        alpha = float(alpha_str)
        norms = np.array(alpha_data["norm"])
        test_errors = np.array(alpha_data["test_errors"])
        train_errors = np.array(alpha_data["train_errors"])
        
        # Compute statistics
        alphas.append(alpha)
        norm_means.append(np.mean(norms))
        norm_stds.append(np.std(norms))
        test_means.append(np.mean(test_errors))
        test_stds.append(np.std(test_errors))
        train_means.append(np.mean(train_errors))
        train_stds.append(np.std(train_errors))
    
    # Sort everything by alpha values
    sorted_indices = np.argsort(alphas)
    alphas = np.array(alphas)[sorted_indices]
    norm_means = np.array(norm_means)[sorted_indices]
    norm_stds = np.array(norm_stds)[sorted_indices]
    test_means = np.array(test_means)[sorted_indices]
    test_stds = np.array(test_stds)[sorted_indices]
    train_means = np.array(train_means)[sorted_indices]
    train_stds = np.array(train_stds)[sorted_indices]
    
    return alphas, norm_means, norm_stds, test_means, test_stds, train_means, train_stds




def read_emp_esd_results(k, k_0, alpha, R_00, lambda_reg=0, d=250):
    print(f"Reading empirical ESD results for k={k}, k_0={k_0}, alpha={alpha}, R_00={R_00}, lambda_reg={lambda_reg}, d={d}")
    # Find matching file
    data_dir = os.path.join(os.path.dirname(__file__), "data", "mle_empirical")
    data_filename = f"esd_data_k{k}_k0{k_0}_lambda{lambda_reg}_d{d}_alpha{alpha:.2f}.json"
    data_filepath = os.path.join(data_dir, data_filename)
    
    # Read the data file
    with open(data_filepath, 'r') as f:
        closest_data = json.load(f)
    
    # Convert eigenvalues back to numpy array
    eigenvalues = np.array(closest_data["eigenvalues"])
    actual_alpha = closest_data["metadata"]['alpha']
    
    if len(eigenvalues.shape) == 3:  # If eigenvalues is 3D array
        avg_eigenvalues = np.mean(eigenvalues, axis=0).flatten()
    else:  # If eigenvalues is already 2D
        avg_eigenvalues = np.mean(eigenvalues, axis=0)

    print('avg eigenvalues shape:', avg_eigenvalues.shape)
    print('eigenvalues shape:', eigenvalues.shape)
    return avg_eigenvalues, eigenvalues, closest_data["metadata"]['R_00'], closest_data["metadata"]['alpha']



def read_mle_esd(k, k_0, alpha, d=250, n_trials=100):
    # Construct filepath
    data_dir = os.path.join(os.path.dirname(__file__), "newdata", "mle_empirical", "esd")
    filename = f"esd_k{k}_k0{k_0}_d{d}_ntrials{n_trials}.json"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"No ESD data file found at: {filepath}")
        return None
    
    # Read the file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert alpha to string for dictionary lookup
    alpha_str = str(int(alpha))  # Currently only supports alpha=3 or alpha=5
    
    if alpha_str not in data["results"]:
        print(f"No ESD data found for alpha={alpha}")
        return None
    
    # Convert back to numpy array
    esd_data = np.array(data["results"][alpha_str])
    
    return esd_data








def get_mle_regularized_results(k_0, k, R_00, d, type_3, n_trials):

    # Create base filename
    if type_3 == False:
        data_dir = os.path.join(os.path.dirname(__file__), "newdata", "mle_empirical")
        base_filename = f"mle_reg_k{k}_k0{k_0}_d{d}_ntrials{n_trials}.json"
    elif type_3 != False:
        data_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "mle_empirical")
        base_filename = f"MLE_reg_evals_(k={k},k0={k_0})_{type_3}.json"
    filepath = os.path.join(data_dir, base_filename)

    with open(filepath, 'r') as f:
        data = json.load(f)
        
    results = {}
    
    # Process all results without filtering by R_00
    for result_key in data["results"].keys():
        # Try to parse the key as JSON
        result_dict = None
        if isinstance(result_key, dict):
            result_dict = result_key
        elif isinstance(result_key, str):
            try:
                result_dict = json.loads(result_key)
            except (json.JSONDecodeError, TypeError):
                # Skip keys that aren't valid JSON strings with the expected structure
                continue
        
        # Check if result_dict has the expected structure
        if not isinstance(result_dict, dict) or "alpha" not in result_dict or "lambda_reg" not in result_dict:
            continue
            
        alpha = result_dict["alpha"]
        lambda_reg = result_dict["lambda_reg"]
        metrics = data["results"][result_key]
        
        if alpha not in results:
            results[alpha] = {}
            
        results[alpha][lambda_reg] = {
            'test_errors': metrics['test_errors'],
            'train_errors': metrics['train_errors'],
            'norms': metrics['norms'],
            'misclassification_test_errors': metrics['misclassification_test_errors']
        }
                    
    return results


