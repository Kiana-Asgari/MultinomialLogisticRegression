import matplotlib.pyplot as plt
import numpy as np
import os
from multinomial_logistic.log_data.log_fp_tests import read_fp_test_results
from multinomial_logistic.log_data.log_mle_empirical import read_mle_results

def collect_empirical_data(k, k_0, lambda_reg, R_00_values, alphas, window_size=0.5):
    """
    Collect empirical data for F-norm from MLE results, filtering alphas by window_size.
    
    Args:
        k, k_0, lambda_reg: System parameters
        R_00_values: List of R_00 matrices
        alphas: List of alpha values
        window_size (float): Minimum distance between selected alpha values
    
    Returns:
        dict: Dictionary containing filtered empirical data for each R_00 value
    """
    # First collect all data
    full_data = {str(R_00.tolist()): {'alphas': [], 'norms': []} for R_00 in R_00_values}
    
    for R_00 in R_00_values:
        R_00_key = str(R_00.tolist())
        for alpha in alphas:
            results = read_mle_results(alpha, k, k_0, R_00, lambda_reg)
            if results is None or results[4]:  # Skip if None or diverged
                continue
            norms, _, _, _, _ = results
            
            full_data[R_00_key]['alphas'].append(alpha)
            full_data[R_00_key]['norms'].append(norms)
    
    # Filter data based on window_size
    filtered_data = {str(R_00.tolist()): {'alphas': [], 'norms': []} for R_00 in R_00_values}
    
    for R_00 in R_00_values:
        R_00_key = str(R_00.tolist())
        if not full_data[R_00_key]['alphas']:  # Skip if no data
            continue
            
        # Sort data by alpha values
        sorted_indices = np.argsort(full_data[R_00_key]['alphas'])
        sorted_alphas = np.array(full_data[R_00_key]['alphas'])[sorted_indices]
        sorted_norms = np.array(full_data[R_00_key]['norms'])[sorted_indices]
        
        # Always include the first point
        last_included_alpha = sorted_alphas[0]
        filtered_data[R_00_key]['alphas'].append(sorted_alphas[0])
        filtered_data[R_00_key]['norms'].append(sorted_norms[0])
        
        # Filter points based on window_size
        for i in range(1, len(sorted_alphas)):
            if sorted_alphas[i] - last_included_alpha >= window_size:
                filtered_data[R_00_key]['alphas'].append(sorted_alphas[i])
                filtered_data[R_00_key]['norms'].append(sorted_norms[i])
                last_included_alpha = sorted_alphas[i]
        
        # Always include the last point if it wasn't already included
        if filtered_data[R_00_key]['alphas'][-1] != sorted_alphas[-1]:
            filtered_data[R_00_key]['alphas'].append(sorted_alphas[-1])
            filtered_data[R_00_key]['norms'].append(sorted_norms[-1])
    
    return filtered_data

def plot_fp_test_results(k, k_0, lambda_reg=0, empirical_mean_std=False, empirical_median_quantile=False, 
                        empirical_mean_only=False, window_size=1, alpha_min=2.8):
    """
    Creates three plots: test error, train error, and F-norm vs alpha.
    Each plot contains multiple curves for different R_00 values.
    
    Args:
        window_size (float): Minimum distance between empirical alpha values to plot
        alpha_min (float): Minimum alpha value to plot
        empirical_mean_only (bool): If True, only plot mean without std bands
    """
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Define R_00 values and alphas
    R_00_values = [np.eye(k), np.array([[1,0.5], [0.5,1]])]
    alphas = np.concatenate([np.linspace(2.8, 5, 20), np.linspace(5, 20, 30)]).flatten()
    alphas = alphas[alphas >= alpha_min]  # Filter alphas by alpha_min
    alphas = np.sort(alphas)[::-1]
    
    # Initialize data storage
    test_errors = {str(R_00.tolist()): [] for R_00 in R_00_values}
    train_errors = {str(R_00.tolist()): [] for R_00 in R_00_values}
    f_norms = {str(R_00.tolist()): [] for R_00 in R_00_values}
    alpha_values = {str(R_00.tolist()): [] for R_00 in R_00_values}
    
    # Collect theoretical data
    for R_00 in R_00_values:
        R_00_key = str(R_00.tolist())
        for alpha in alphas:
            results = read_fp_test_results(alpha, k, k_0, R_00, lambda_reg)
            if results is None or results[4]:  # Skip if None or diverged
                continue
            test_err, train_err, f_norm, _, _ = results
            
            test_errors[R_00_key].append(test_err)
            train_errors[R_00_key].append(train_err)
            f_norms[R_00_key].append(f_norm)
            alpha_values[R_00_key].append(alpha)
    
    # Collect empirical data if requested
    empirical_data = None
    if empirical_mean_std or empirical_median_quantile:
        empirical_data = collect_empirical_data(k, k_0, lambda_reg, R_00_values, alphas, window_size)
    
    # Create and save plots
    labels = {str(np.eye(k).tolist()): "R₀₀ = I",
             str([[1,0.5], [0.5,1]]): "R₀₀ = [[1,0.5],[0.5,1]]"}
    
    # Plot test errors
    plt.figure(figsize=(10, 6))
    for R_00 in R_00_values:
        R_00_key = str(R_00.tolist())
        plt.plot(alpha_values[R_00_key], test_errors[R_00_key], 'o-', label=labels[R_00_key])
    plt.xlabel('α')
    plt.ylabel('Test Error')
    plt.title(f'Test Error vs α (k={k}, k₀={k_0}, λ={lambda_reg})')
    plt.legend()
    plt.grid(True)
    
    # Save test error plot
    base_name = f"test_error_k{k}_k0{k_0}_lambda{lambda_reg}"
    counter = 0
    while os.path.exists(os.path.join(figures_dir, f"{base_name}_{counter}.png")):
        counter += 1
    plt.savefig(os.path.join(figures_dir, f"{base_name}_{counter}.png"))
    plt.close()
    
    # Plot train errors (similar to test errors)
    plt.figure(figsize=(10, 6))
    for R_00 in R_00_values:
        R_00_key = str(R_00.tolist())
        plt.plot(alpha_values[R_00_key], train_errors[R_00_key], 'o-', label=labels[R_00_key])
    plt.xlabel('α')
    plt.ylabel('Train Error')
    plt.title(f'Train Error vs α (k={k}, k₀={k_0}, λ={lambda_reg})')
    plt.legend()
    plt.grid(True)
    
    # Save train error plot
    base_name = f"train_error_k{k}_k0{k_0}_lambda{lambda_reg}"
    counter = 0
    while os.path.exists(os.path.join(figures_dir, f"{base_name}_{counter}.png")):
        counter += 1
    plt.savefig(os.path.join(figures_dir, f"{base_name}_{counter}.png"))
    plt.close()
    
    # Plot F-norms with empirical data
    plt.figure(figsize=(10, 6))
    for R_00 in R_00_values:
        R_00_key = str(R_00.tolist())
        color = 'C0' if R_00_key == str(np.eye(k).tolist()) else 'C1'
        
        # Plot theoretical F-norm
        plt.plot(alpha_values[R_00_key], f_norms[R_00_key], 'o-', 
                label=f"{labels[R_00_key]} (theoretical)", color=color)
        
        # Add empirical data if available
        if empirical_data and R_00_key in empirical_data:
            emp_alphas = empirical_data[R_00_key]['alphas']
            emp_norms = empirical_data[R_00_key]['norms']
            
            if emp_alphas:
                if empirical_mean_std or empirical_mean_only:
                    means = np.array([np.mean(norms) for norms in emp_norms])
                    if empirical_mean_std and not empirical_mean_only:
                        stds = np.array([np.std(norms) for norms in emp_norms])
                        plt.fill_between(emp_alphas, means - stds, means + stds, alpha=0.3, color=color)
                    plt.plot(emp_alphas, means, '--', label=f"{labels[R_00_key]} (empirical mean)", color=color)
                
                if empirical_median_quantile:
                    medians = np.array([np.median(norms) for norms in emp_norms])
                    q25 = np.array([np.percentile(norms, 25) for norms in emp_norms])
                    q75 = np.array([np.percentile(norms, 75) for norms in emp_norms])
                    plt.fill_between(emp_alphas, q25, q75, alpha=0.3, color='C2' if color == 'C0' else 'C3')
                    plt.plot(emp_alphas, medians, ':', 
                           label=f"{labels[R_00_key]} (empirical median)", 
                           color='C2' if color == 'C0' else 'C3')
    
    plt.xlabel('α')
    plt.ylabel('F-norm')
    title = f'F-norm vs α (k={k}, k₀={k_0}, λ={lambda_reg})'
    if empirical_mean_std and not empirical_mean_only:
        title += '\nwith empirical mean±std'
    elif empirical_mean_only:
        title += '\nwith empirical mean'
    if empirical_median_quantile:
        title += '\nwith empirical median/quartiles'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save F-norm plot with all active flags in the name
    base_name = f"f_norm_k{k}_k0{k_0}_lambda{lambda_reg}"
    
    # Add all active boolean flags to filename
    if empirical_mean_std and not empirical_mean_only:
        base_name += "_mean_std"
    elif empirical_mean_only:
        base_name += "_mean_only"
    if empirical_median_quantile:
        base_name += "_median_quantile"
    
    counter = 0
    while os.path.exists(os.path.join(figures_dir, f"{base_name}_{counter}.png")):
        counter += 1
    plt.savefig(os.path.join(figures_dir, f"{base_name}_{counter}.png"))
    plt.close()
