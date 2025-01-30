import os
import json
import numpy as np
import matplotlib.pyplot as plt
from real_data.eval.fit_data import fit_data

def plot_esd_density(X_train, y_train, X_test, y_test, alpha, file_number=1):
    n_samples = int(alpha * X_train.shape[1])
    esd_values_mnist = []
    n_iter = 5

    for i in range(n_iter):
        np.random.seed(5*i+2)
        sample_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train_sampled = X_train[sample_indices]
        y_train_sampled = y_train[sample_indices]
        results = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=True, seed=i)
        esd_values_mnist.append(results['esd_values'])

    print("esd_values min: ", np.min(esd_values_mnist), "max: ", np.max(esd_values_mnist))
        
    # Create histogram and plot MP distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(esd_values_mnist).flatten(), bins=100, density=True, alpha=0.5,
             color='blue', label=f'Empirical for $\\alpha={alpha}$')
    """
    Reads the ESD Hessian data for a given alpha and plots density vs z_real values.
    
    Args:
        alpha (float): The alpha value to plot
        file_number (int): The file number suffix (default=1)
    """
    # Construct the filepath
    base_filename = f"esd_data_alpha{alpha}_{file_number}.json"
    base_filepath = os.path.join(os.path.dirname(__file__), "..", "eval", "data", "esd_theoretical", base_filename)
    
    if not os.path.exists(base_filepath):
        raise FileNotFoundError(f"No data file found at {base_filepath}")
    
    # Load the data
    with open(base_filepath, 'r') as f:
        data = json.load(f)
    
    results = data["results"]
    
    # Extract z_real values and densities
    z_real_values = []
    densities = []
    
    # There should be only one R_00 key in the results
    R_00_key = list(results.keys())[0]
    alpha_str = str(alpha)
    
    for z_real_str in results[R_00_key][alpha_str].keys():
        z_real = float(z_real_str)
        # Get the first (and should be only) z_imag value
        z_imag_key = list(results[R_00_key][alpha_str][z_real_str].keys())[0]
        density = results[R_00_key][alpha_str][z_real_str][z_imag_key]['density']
        
        z_real_values.append(z_real)
        densities.append(density)
    
    # Sort by z_real values to ensure proper plotting
    sorted_indices = np.argsort(z_real_values)
    z_real_values = np.array(z_real_values)[sorted_indices]
    densities = np.array(densities)[sorted_indices]
    

    plt.plot(z_real_values, densities, color='red', label=f'α={alpha}')
    plt.xlabel('λ')
    plt.ylabel('ρ(λ)')
    plt.title(f'Spectral Density for α={alpha}')
    plt.grid(True)
    plt.legend()
    
    # Optional: save the plot
    plot_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'esd_density_alpha{alpha}.pdf'))
    
    plt.show()
    
    return z_real_values, densities
