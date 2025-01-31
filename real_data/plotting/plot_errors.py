import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_errors_comparison(feature_name, n_hidden, file_number=1):
    if n_hidden == 500:
        emp_min_alpha = 4.5
        theory_min_alpha = 0
    else:
        emp_min_alpha = 0
        theory_min_alpha = 0

    # Set up plotting style
    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7}) 
    plt.rcParams.update({
        'text.usetex': True,            # For LaTeX rendering
        'font.family': 'serif',         # Use serif font family
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',       # Use Computer Modern math font
        'figure.dpi': 120,              
        'figure.figsize': (7, 5),      
        'font.size': 18, 
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}', # Added bm package
        'mathtext.default': 'regular',   # Use regular (serif) font for math
        'axes.formatter.use_mathtext': True,  # Use mathtext for axis formatting
    })

    # Load empirical data
    emp_filename = f"error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    emp_filepath = os.path.join("real_data", "eval", "data", "error_empirical", emp_filename)
    
    # Load theoretical data
    theory_filename = f"error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    theory_filepath = os.path.join("real_data", "eval", "data", "error_theoretical", theory_filename)

    # Check if files exist
    if not os.path.exists(emp_filepath) or not os.path.exists(theory_filepath):
        raise FileNotFoundError("One or both data files not found")

    # Load the data
    with open(emp_filepath, 'r') as f:
        emp_data = json.load(f)
    with open(theory_filepath, 'r') as f:
        theory_data = json.load(f)

    # Extract data for plotting
    emp_results = emp_data["results"]
    theory_results = theory_data["results"]
    
    # Get the R_00 key from theoretical results
    R_00_key = list(theory_results.keys())[0]
    print('R_00_key', R_00_key)

    # Create lists to store the data
    alphas_emp = []
    train_errors_emp = []
    test_errors_emp = []
    class_errors_emp = []
    train_std_emp = []
    test_std_emp = []
    class_std_emp = []
    
    alphas_theory = []
    train_errors_theory = []
    test_errors_theory = []
    class_errors_theory = []

    # Extract empirical data with standard errors
    for alpha_str, result in emp_results.items():
        alpha = float(alpha_str)
        alphas_emp.append(alpha)
        train_errors_emp.append(result["mean_train_error"])
        test_errors_emp.append(result["mean_test_error"])
        class_errors_emp.append(result["mean_classification_error"])
        
        # Extract standard errors (standard deviation divided by sqrt of n_iter)
        train_std_emp.append(result["std_train_error"] / np.sqrt(result.get("n_iter", 1)))
        test_std_emp.append(result["std_test_error"] / np.sqrt(result.get("n_iter", 1)))
        class_std_emp.append(result["std_classification_error"] / np.sqrt(result.get("n_iter", 1)))

    # Filter data for alphas > 5
    mask_emp = np.array(alphas_emp) > emp_min_alpha
    alphas_emp = np.array(alphas_emp)[mask_emp]
    train_errors_emp = np.array(train_errors_emp)[mask_emp]
    test_errors_emp = np.array(test_errors_emp)[mask_emp]
    class_errors_emp = np.array(class_errors_emp)[mask_emp]
    train_std_emp = np.array(train_std_emp)[mask_emp]
    test_std_emp = np.array(test_std_emp)[mask_emp]
    class_std_emp = np.array(class_std_emp)[mask_emp]

    # Convert to numpy arrays for easier plotting
    alphas_emp = np.array(alphas_emp)
    train_errors_emp = np.array(train_errors_emp)
    test_errors_emp = np.array(test_errors_emp)
    class_errors_emp = np.array(class_errors_emp)
    train_std_emp = np.array(train_std_emp)
    test_std_emp = np.array(test_std_emp)
    class_std_emp = np.array(class_std_emp)
    n_iter = 50

    # Extract theoretical data
    for alpha_str, result in theory_results[R_00_key].items():
        alpha = float(alpha_str)
        alphas_theory.append(alpha)
        train_errors_theory.append(result["train_error"])
        test_errors_theory.append(result["test_error"])
        class_errors_theory.append(result["misclassification_test_error"])

    # Convert theoretical data to numpy arrays and filter
    alphas_theory = np.array(alphas_theory)
    train_errors_theory = np.array(train_errors_theory)
    test_errors_theory = np.array(test_errors_theory)
    class_errors_theory = np.array(class_errors_theory)
    
    mask_theory = alphas_theory > theory_min_alpha
    alphas_theory = alphas_theory[mask_theory]
    train_errors_theory = train_errors_theory[mask_theory]
    test_errors_theory = test_errors_theory[mask_theory]
    class_errors_theory = class_errors_theory[mask_theory]

    # Create separate figures for each error type
    # Train Error
    plt.figure(figsize=(8, 6))
    
    # Filter train error data within range [0.2, 0.5]
    filtered_alphas_emp, filtered_train_emp, filtered_train_std = filter_data_by_range(
        alphas_emp, train_errors_emp, 0.2, 0.55, train_std_emp)
    filtered_alphas_theory, filtered_train_theory, _ = filter_data_by_range(
        alphas_theory, train_errors_theory, 0.2, 0.55)
    
    plt.errorbar(filtered_alphas_emp, filtered_train_emp, 
                yerr=filtered_train_std/np.sqrt(n_iter), color='blue',
                fmt='o', alpha=0.6, capsize=2, markersize=2.5)
    plt.plot(filtered_alphas_theory, filtered_train_theory, '-', 
            alpha=0.7, color='darkblue')
    plt.xlabel(r'$\alpha$')
    plt.ylim(0.19, 0.56)
    plt.ylabel('train error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save train error plot
    save_dir = os.path.join("real_data", "plotting", "figures", feature_name, f"{n_hidden}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'train_{feature_name}_{n_hidden}.pdf'))
    plt.close()

    # Test Error
    plt.figure(figsize=(8, 6))
    
    # Filter test error data within range [0.5, 1.5]
    filtered_alphas_emp, filtered_test_emp, filtered_test_std = filter_data_by_range(
        alphas_emp, test_errors_emp, 0.55, 1.5, test_std_emp)
    filtered_alphas_theory, filtered_test_theory, _ = filter_data_by_range(
        alphas_theory, test_errors_theory, 0.55, 1.5)
    
    plt.errorbar(filtered_alphas_emp, filtered_test_emp, 
                yerr=filtered_test_std/np.sqrt(n_iter), color='blue',
                fmt='o', alpha=0.6, capsize=2, markersize=2.5)
    plt.plot(filtered_alphas_theory, filtered_test_theory, '-', 
            alpha=0.7, color='darkblue')
    plt.xlabel(r'$\alpha$')
    plt.ylim(0.55, 1.5)
    plt.ylabel('(log loss) test error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save test error plot
    save_dir = os.path.join("real_data", "plotting", "figures", feature_name, f"{n_hidden}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'test_{feature_name}_{n_hidden}.pdf'))
    plt.close()

    # Classification Error
    plt.figure(figsize=(8, 6))
    
    # Filter classification error data within range [0.24, 0.38]
    filtered_alphas_emp, filtered_class_emp, filtered_class_std = filter_data_by_range(
        alphas_emp, class_errors_emp, 0.24, 0.38, class_std_emp)
    filtered_alphas_theory, filtered_class_theory, _ = filter_data_by_range(
        alphas_theory, class_errors_theory, 0.24, 0.38)
    
    plt.errorbar(filtered_alphas_emp, filtered_class_emp, 
                yerr=filtered_class_std/np.sqrt(n_iter), color='blue',
                fmt='o', alpha=0.6, capsize=2, markersize=2.5)
    plt.plot(filtered_alphas_theory, filtered_class_theory, '-', 
            alpha=0.7, color='darkblue')
    plt.xlabel(r'$\alpha$')
    plt.ylim(0.24, 0.38)
    plt.ylabel('(classification) test error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save classification error plot
    save_dir = os.path.join("real_data", "plotting", "figures", feature_name, f"{n_hidden}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'classification_{feature_name}_{n_hidden}.pdf'))
    plt.close()
    print('plots saved at', save_dir)

def filter_data_by_range(x_data, y_data, min_val, max_val, std_data=None):
    """Helper function to filter data within a specified range."""
    mask = (y_data >= min_val) & (y_data <= max_val)
    filtered_x = x_data[mask]
    filtered_y = y_data[mask]
    filtered_std = std_data[mask] if std_data is not None else None
    return filtered_x, filtered_y, filtered_std
