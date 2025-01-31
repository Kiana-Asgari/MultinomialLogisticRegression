import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bays(feature_name, n_hidden, file_number=1):

    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 14
    })

    # Load empirical data
    emp_filename = f"error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    emp_filepath = os.path.join("real_data", "eval", "data", "error_empirical", emp_filename)
    
    # Load theoretical data
    theory_filename = f"error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    theory_filepath = os.path.join("real_data", "eval", "data", "error_theoretical", theory_filename)

    irr_filename = f"irreducible_error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    irr_filepath = os.path.join("real_data", "eval", "data", "error_theoretical", irr_filename)

    bay_filename = f"bayesian_error_data_feature_name={feature_name}_n_hidden={n_hidden}_file_number={file_number}.json"
    bay_filepath = os.path.join("real_data", "eval", "data", "error_empirical", bay_filename)

    # Check if files exist
    if not os.path.exists(emp_filepath) or not os.path.exists(theory_filepath):
        raise FileNotFoundError("One or both data files not found")

    # Load the data
    with open(emp_filepath, 'r') as f:
        emp_data = json.load(f)
    with open(theory_filepath, 'r') as f:
        theory_data = json.load(f)

    with open(irr_filepath, 'r') as f:
        irr_data = json.load(f)

    with open(bay_filepath, 'r') as f:
        bay_data = json.load(f)

    # Extract data for plotting
    emp_results = emp_data["results"]
    theory_results = theory_data["results"]
    
    # Get the R_00 key from theoretical results
    R_00_key = list(theory_results.keys())[0]

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
    mask_emp = np.array(alphas_emp) > 5
    alphas_emp = np.array(alphas_emp)[mask_emp]
    test_errors_emp_normalized = np.array(test_errors_emp)[mask_emp] - bay_data["results"]["mean_test_error"]
    class_errors_emp_normalized = np.array(class_errors_emp)[mask_emp] - bay_data["results"]["mean_classification_error"]

    test_std_emp = np.array(test_std_emp)[mask_emp]
    class_std_emp = np.array(class_std_emp)[mask_emp]

    # Convert to numpy arrays for easier plotting
    alphas_emp = np.array(alphas_emp)
    test_errors_emp_normalized = np.array(test_errors_emp_normalized)
    class_errors_emp_normalized = np.array(class_errors_emp_normalized)

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

    # Get the first (and only) R_00 key from irreducible error data
    irr_R_00_key = list(irr_data["results"].keys())[0]
    irr_test = irr_data["results"][irr_R_00_key]["irreducible_test_error"]
    irr_misclass = irr_data["results"][irr_R_00_key]["irreducible_misclassification_error"]

    # Convert theoretical data to numpy arrays and filter
    alphas_theory = np.array(alphas_theory)
    test_errors_theory_normalized = np.array(test_errors_theory) - irr_test
    class_errors_theory_normalized = np.array(class_errors_theory) - irr_misclass
    
    mask_theory = alphas_theory > 5
    alphas_theory = alphas_theory[mask_theory]
    test_errors_theory_normalized = test_errors_theory_normalized[mask_theory]
    class_errors_theory_normalized = class_errors_theory_normalized[mask_theory]

    # Create three subplots
    fig, ( ax2, ax3) = plt.subplots(1,2, figsize=(12, 6))
    

    # Plot test error with error bars
    ax2.errorbar(alphas_emp, test_errors_emp_normalized, yerr=test_std_emp/np.sqrt(n_iter), color='blue',
                fmt='o', alpha=0.7, label='Empirical', capsize=2, markersize=3)
    ax2.plot(alphas_theory, test_errors_theory_normalized, '-', label='Theory', alpha=0.7, color='darkblue')
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Test Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot classification error with error bars
    ax3.errorbar(alphas_emp, class_errors_emp_normalized, yerr=class_std_emp/np.sqrt(n_iter), color='blue',
                fmt='o', alpha=0.7, label='Empirical', capsize=2, markersize=3)
    ax3.plot(alphas_theory, class_errors_theory_normalized, '-', label='Theory', alpha=0.7, color='darkblue')
    ax3.set_xlabel(r'$\alpha$')
    ax3.set_ylabel('Classification Error')
    ax3.set_ylim(0., 0.12)  # Set y-axis limits
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join("real_data", "plotting", "figures")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'error_bayesian_comparison_{feature_name}_n_hidden_{n_hidden}.pdf'))
    plt.close()
