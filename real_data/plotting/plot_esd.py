import os
import json
import numpy as np
import matplotlib.pyplot as plt
from real_data.eval.fit_data import fit_data
import matplotlib as mpl
import seaborn as sns

def plot_esd_density(X_train, y_train, X_test, y_test, alpha, file_number=1):

    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7}) 
    mpl.rcParams.update({
        'text.usetex': True,            # For LaTeX rendering
        'font.family': 'serif',         # Use serif font family
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',       # Use Computer Modern math font
        'figure.dpi': 120,              
        'figure.figsize': (7, 5),       
        'axes.labelsize': 16,           
        'axes.titlesize': 16,           
        'xtick.labelsize': 14,          
        'ytick.labelsize': 14,          
        'legend.fontsize': 14,          
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'font.size': 14,                
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}', # Added bm package
        'mathtext.default': 'regular',   # Use regular (serif) font for math
        'axes.formatter.use_mathtext': True,  # Use mathtext for axis formatting
    })
    fig, ax = plt.subplots()




    n_samples = int(alpha * X_train.shape[1])
    esd_values_mnist = []
    n_iter = 25

    for i in range(n_iter):
        np.random.seed(5*i+2)
        sample_indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_train_sampled = X_train[sample_indices]
        y_train_sampled = y_train[sample_indices]
        results = fit_data(X_train_sampled, y_train_sampled, X_test=X_test, y_test=y_test, compute_esd=True, seed=i)
        esd_values_mnist.append(results['esd_values'])
        print('iter', i, 'done')
    print("esd_values min: ", np.min(esd_values_mnist), "max: ", np.max(esd_values_mnist))
        
    # Create histogram and plot MP distribution
        
    sns.histplot(
        data=np.array(esd_values_mnist).flatten(),
        bins=100,
        stat='density',
        fill=True,
        alpha=0.5,
        #linewidth=1,
        color='blue',
        label='Empirical spectrum on MNIST dataset',
        ax=ax
    )

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

    print('available theoretical values:')
    for i in range(len(z_real_values)):
        print('z_real: ', z_real_values[i], 'density: ', densities[i])
    

    if alpha == 10.0:
        z_real_values = z_real_values[3:-17]
        densities = densities[3:-17]
        densities[0] = 0.0
        densities[-1] = 0.0
    if alpha == 30.0:
        z_real_values = z_real_values[5:-55]
        densities = densities[5:-55]
        densities[0] = 0.0
        z_real_values[-1] = z_real_values[-2] + 2*1e-3
        densities[-1] = 0.0





    ax.plot(
        z_real_values, 
        densities, 
        color='darkblue', 
        linewidth=2.4, 
        label=r'$\mu_{\star}(\nu_{\mathrm{opt}})$'
    )
    ax.set_xlabel(r'$\lambda$')
    ax.set_xlim(0, 0.35)
    ax.set_ylabel('')
    ax.set_title(f'Spectral Density for $\\alpha={alpha}$')
    ax.grid(True)
    ax.legend()
    
    # Optional: save the plot
    plot_dir = os.path.join(os.path.dirname(__file__), "figures", "esd")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'esd_350_tanh_alpha{alpha}.pdf'))
    
    plt.show()
    
    return z_real_values, densities

