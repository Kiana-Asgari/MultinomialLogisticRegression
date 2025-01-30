import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.evaluation.misclassification_test_error import misclassification_test_error
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from multinomial_logistic.MLE_empirical.mle_empirical_skitlearn import _batched_hessian


def edit_theoretical_errors(R_00, k, k_0, n_hidden,
                                     feature_name='ReLU', name_data="fashion_mnist",
                                     classes_to_keep=[2,4,6], tol=1e-5, max_iter=200):
    # Define the directory and filename for saving results
    print('  editing results for n_hidden = ', n_hidden, 'R_00 = ', R_00)
    full_dir = os.path.join("mnist_test", "log", "new_fp+errors")
    os.makedirs(full_dir, exist_ok=True)  # Create all necessary subdirectories
    results_file = os.path.join(full_dir, f"fp_n_hidden_{n_hidden}_data_{name_data}_feature_{feature_name}.json")

    # Load existing results if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}


    schur_0 = R_00
    R_01_0 = np.zeros((k, k_0))
    S_0 = np.eye(k)
    alpha_list = np.concatenate((np.linspace(12, 5, num=40), np.linspace(5, 3.5, num=5)))
    alphas_available = list(results.keys())
    print('alphas_available: ', alphas_available)

    # Iterate over alpha values
    for alpha_str in alphas_available:  # Adjust num for more granularity if needed
        print('alpha_str: ', alpha_str)
        alpha = float(alpha_str)
        #if alpha>12:
        #    continue

        # Check if this alpha and n_hidden combination is already logged
        print(f" alpha={alpha} exists. editing the theoretical errors")
        schur_0 = np.array(results[alpha_str]["schur_t"])
        R_01_0 = np.array(results[alpha_str]["R_01_t"])
        S_0 = np.array(results[alpha_str]["S_t"])

        #mle_test_errors = np.array(results[alpha_str]["mle_test_errors"])
        #mle_train_errors = np.array(results[alpha_str]["mle_train_errors"])
        #mle_misclass_test_errors = np.array(results[alpha_str]["mle_misclass_test_errors"])
        #mle_f_norms = np.array(results[alpha_str]["mle_f_norms"])

            
        if alpha < 5:
            tol = 1e-2
        else:
            tol = 1e-4

        # Run the state evolution recursion
        print(' continuing the state evolution with the logged data')
        schur_0, R_01_0, S_0, divergence = state_evolution_full_recursion(
                        R_00=R_00, schur_0=schur_0, R_01_0=R_01_0,
                        alpha=alpha, k=k, k_0=k_0, lambda_reg=0, S_0=S_0,
                        tol=tol, max_iter=max_iter
        )
        A = R_01_0 @ np.linalg.inv(sqrtm(R_00))
        test_error_theoretical = test_error(R_00=R_00, schur=schur_0, R_01=R_01_0, k=k, k_0=k_0, alpha=alpha)
        misclass_test_error_theoretical = misclassification_test_error(S=None, R_00=R_00, schur_t=schur_0, A_t=A, alpha=alpha, k=k, k_0=k_0)
        train_error_theoretical = train_error(R_00=R_00, schur=schur_0, R_01=R_01_0, S=S_0, alpha=alpha, k=k, k_0=k_0)

      
        # Save the results




        results[alpha] = {
            "schur_t": schur_0.tolist(),
            "R_01_t": R_01_0.tolist(),
            "S_t": S_0.tolist(),
            "divergence": divergence,
            "R_00": R_00.tolist(),
            "n_hidden": n_hidden,
            "name_data": name_data,
            "classes_to_keep": classes_to_keep,
            "missclass_test_error": misclass_test_error_theoretical.tolist(),
            "train_error": train_error_theoretical.tolist(),
            "test_error": test_error_theoretical.tolist(),
            "new_computation": True
           # "mle_test_errors": mle_test_errors.tolist(),
           # "mle_train_errors": mle_train_errors.tolist(),
           # "mle_misclass_test_errors": mle_misclass_test_errors.tolist(),
           # "mle_f_norms": mle_f_norms.tolist()
        }

        # Write results to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print( 'theoretical misclass: ', misclass_test_error_theoretical)
        print(   'theoretical train: ', train_error_theoretical)
        print( 'theoretical test: ', test_error_theoretical)
        print(f"Results for alpha={alpha_str} saved.")







def run_mle_and_save(R_00, Theta_0, k, k_0, n_hidden,
                                     X_train, y_train, X_test, y_test,
                                     n_iter=10, y_train_full=None, y_test_full=None,
                                     feature_name='ReLU', name_data="fashion_mnist",
                                     classes_to_keep=[2,4,6], tol=1e-5, max_iter=200):
    # Define the directory and filename for saving results
    print('  logging the fp solution for n_hidden = ', n_hidden, 'R_00 = ', R_00)
    full_dir = os.path.join("mnist_test", "log", "fp+errors")
    os.makedirs(full_dir, exist_ok=True)  # Create all necessary subdirectories
    results_file = os.path.join(full_dir, f"mle_n_hidden_{n_hidden}_data_{name_data}_feature_{feature_name}.json")

    # Load existing results if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}


    alpha_list = (np.linspace(20, 5, num=40)).flatten()[::-1]

    # Iterate over alpha values
    for alpha in alpha_list:  # Adjust num for more granularity if needed
        alpha_str = f"{alpha}"
        
        # Check if this alpha and n_hidden combination is already logged
        if alpha_str in results and results[alpha_str].get("n_hidden") == n_hidden and results[alpha_str].get("classes_to_keep") == classes_to_keep:
            print(f"Skipping alpha={alpha_str}, n_hidden={n_hidden}, classes_to_keep={classes_to_keep} as this combination is already logged.")
            continue
        if alpha < 5:
            tol = 1
        else:
            tol = 1e-4
        mle_test_errors, mle_train_errors, mle_misclass_test_errors, mle_f_norms = evaluate_mle(alpha=alpha, n_hidden=n_hidden,\
                                                R_00=R_00, Theta_0=Theta_0, k=k, k_0=k_0, \
                                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, \
                                                n_iter=n_iter, tol=tol, y_train_full=y_train_full, y_test_full=y_test_full, plot_esd=False)


      
        # Save the results


        results[alpha_str] = {
            "R_00": R_00.tolist(),
            "n_hidden": n_hidden,
            "name_data": name_data,
            "classes_to_keep": classes_to_keep,
            "mle_test_errors": mle_test_errors.tolist(),
            "mle_train_errors": mle_train_errors.tolist(),
            "mle_misclass_test_errors": mle_misclass_test_errors.tolist(),
            "mle_f_norms": mle_f_norms.tolist()
        }

        # Write results to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print('**the mle mean misclass: ', np.mean(mle_misclass_test_errors))
        print('**the mle mean train: ', np.mean(mle_train_errors))
        print('**the mle mean test: ', np.mean(mle_test_errors))
        print(f"Results for alpha={alpha_str} saved.")



def run_state_evolution_and_save(R_00, k, k_0, n_hidden,
                                     feature_name='ReLU', name_data="fashion_mnist",
                                     classes_to_keep=[2,4,6], tol=1e-5, max_iter=200):
    # Define the directory and filename for saving results
    print('  logging the fp solution for n_hidden = ', n_hidden, 'R_00 = ', R_00)
    full_dir = os.path.join("mnist_test", "log", "new_fp+errors")
    os.makedirs(full_dir, exist_ok=True)  # Create all necessary subdirectories
    results_file = os.path.join(full_dir, f"fp_n_hidden_{n_hidden}_data_{name_data}_feature_{feature_name}.json")

    # Load existing results if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}


    schur_0 = R_00
    R_01_0 = np.zeros((k, k_0))
    S_0 = np.eye(k)
    alpha_list = np.concatenate((np.linspace(20, 9, num=20), np.linspace(9, 4.5, num=30)))
    alpha_list = np.sort(alpha_list)[::-1]


    # Iterate over alpha values
    for alpha in alpha_list:  # Adjust num for more granularity if needed
        alpha_str = f"{alpha}"
        
        # Check if this alpha and n_hidden combination is already logged
        if alpha_str in results and results[alpha_str].get("n_hidden") == n_hidden and results[alpha_str].get("classes_to_keep") == classes_to_keep:
            print(f"Skipping alpha={alpha_str}, n_hidden={n_hidden}, classes_to_keep={classes_to_keep} as this combination is already logged.")
            continue
        if alpha < 5:
            tol = 1e-4
        else:
            tol = 1e-4
        # Run the state evolution recursion
        schur_0, R_01_0, S_0, divergence = state_evolution_full_recursion(
                        R_00=R_00, schur_0=schur_0, R_01_0=R_01_0,
                        alpha=alpha, k=k, k_0=k_0, lambda_reg=0, S_0=S_0,
                        tol=tol, max_iter=max_iter
        )
        A = R_01_0 @ np.linalg.inv(sqrtm(R_00))
        test_error_theoretical = test_error(R_00=R_00, schur=schur_0, R_01=R_01_0, k=k, k_0=k_0, alpha=alpha)
        misclass_test_error_theoretical = misclassification_test_error(S=None, R_00=R_00, schur_t=schur_0, A_t=A, alpha=alpha, k=k, k_0=k_0)
        train_error_theoretical = train_error(R_00=R_00, schur=schur_0, R_01=R_01_0, S=S_0, alpha=alpha, k=k, k_0=k_0)

      
        # Save the results




        results[alpha_str] = {
            "schur_t": schur_0.tolist(),
            "R_01_t": R_01_0.tolist(),
            "S_t": S_0.tolist(),
            "divergence": divergence,
            "R_00": R_00.tolist(),
            "n_hidden": n_hidden,
            "name_data": name_data,
            "classes_to_keep": classes_to_keep,
            "missclass_test_error": misclass_test_error_theoretical.tolist(),
            "train_error": train_error_theoretical.tolist(),
            "test_error": test_error_theoretical.tolist(),
        }

        # Write results to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print('theoretical misclass: ', misclass_test_error_theoretical)
        print( 'theoretical train: ', train_error)
        print('theoretical test: ', test_error_theoretical)
        print(f"Results for alpha={alpha_str} saved.")







def evaluate_mle(alpha, n_hidden, Theta_0, R_00, k, k_0, X_train,
                  y_train, X_test, y_test, n_iter=10, tol=1e-2, 
                  y_train_full=None, y_test_full=None, plot_esd=True):
    

    test_errors = []
    misclass_test_errors = []
    train_errors = []
    f_norms = []
    eigvals = []
    # Determine the number of training samples to use
    n_samples = int(alpha * n_hidden)
        
    # Ensure we don't exceed the available training data
    n_samples = min(n_samples, X_train.shape[0])

    print(f"train and evaluate logistic regression with alpha: {alpha}, n_hidden: {n_hidden}, n_iter: {n_iter}")
    print(f" full data size: {X_train.shape[0]}, training on {n_samples} samples")

    for i in range(n_iter):
        # Randomly select n_samples from the training data
        np.random.seed(i)  # Use a different seed each time for variability
        indices = np.random.choice(X_train.shape[0], n_samples, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        y_train_full_subset = y_train_full[indices]
        


        logreg = LogisticRegression(           
                fit_intercept=False,
                penalty=None,
                solver='lbfgs',       # can also use 'sag' or 'saga' if data is large
                max_iter=10000,
        )   

        logreg.fit(X_train_subset, y_train_full_subset)

        Theta_hat = logreg.coef_
        Theta_hat = (Theta_hat - Theta_hat[0])[1:,:]

        avg_test_error = log_loss(y_test_full, logreg.predict_proba(X_test))
        train_error = log_loss(y_train_full_subset, logreg.predict_proba(X_train_subset))
        misclass_test_error = 1 - accuracy_score(y_test_full, logreg.predict(X_test))
        f_norm = np.linalg.norm(Theta_hat - Theta_0)

        test_errors.append(avg_test_error)
        misclass_test_errors.append(misclass_test_error)
        train_errors.append(train_error)
        f_norms.append(f_norm)

        if plot_esd:
            eigvals.append(np.linalg.eigvals(_batched_hessian(Theta_hat, X_train)))

    if plot_esd:
        return np.array(test_errors), np.array(train_errors), np.array(misclass_test_errors), np.array(f_norms), np.array(eigvals)  
    else:
        return np.array(test_errors), np.array(train_errors), np.array(misclass_test_errors), np.array(f_norms)






def read_state_evolution_results(n_hidden, R_00, alpha, k=2, k_0=2, results_dir="state_evolution_results"):
    results_file = os.path.join(results_dir, f"results_n_hidden_{n_hidden}.json")
    # Correct the path to include the subdirectories
    results_file = os.path.join(results_dir, "mnist_test", "log", "fp", f"results_n_hidden_{n_hidden}.json")
            
    if not os.path.exists(results_file):
        print(f"No results file found for n_hidden={n_hidden}.")
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    closest_alpha = None
    closest_R_00_diff = float('inf')
    closest_values = None

    for alpha_str, values in results.items():
        current_alpha = float(alpha_str)
        current_R_00 = np.array(values["R_00"])
        
                # Calculate the difference in R_00
        R_00_diff = np.linalg.norm(current_R_00 - R_00)

        # Check if this is the closest match
        if abs(current_alpha - alpha) < 1e-2 and R_00_diff < closest_R_00_diff:
            closest_alpha = current_alpha
            closest_R_00_diff = R_00_diff
            closest_values = values

    if closest_values is not None:
        S = np.array(closest_values["S_t"]).reshape(k, k)
        R_01 = np.array(closest_values["R_01_t"]).reshape(k, k_0)
        schur = np.array(closest_values["schur_t"]).reshape(k, k)  # Assuming R_11 is stored as schur_t
        R_00 = np.array(closest_values["R_00"]).reshape(k, k)
        alpha = closest_alpha
        print(f"Closest match found for alpha={closest_alpha}.")
        return S, R_01, schur, R_00, alpha
    else:
        print("[WARNING] ***No close match found for fp solution.")
        return None






######################################################
# Plotting functions
######################################################
import seaborn as sns
import matplotlib as mpl

def plot_errors_vs_alpha(n_hidden, name_data="fashion_mnist", 
                         feature_name="ReLU", classes_to_keep=[2,4,6], alpha_min=4.5):
    """
    Reads the fixed point solution data and plots test errors vs alpha.
    """
    # Define the file path
    full_dir = os.path.join("mnist_test", "log", "fp+errors")
    results_file = os.path.join(full_dir, f"fp_n_hidden_{n_hidden}_data_{name_data}_feature_{feature_name}.json")
    print('plotting test errors vs alpha for ', results_file)
    if not os.path.exists(results_file):
        print(f"No results file found at {results_file}")
        return
    print('plotting test errors vs alpha for ', results_file)

    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7}) 
    mpl.rcParams.update({
        'text.usetex': True,            # For LaTeX rendering
        'font.family': 'serif',         # Use serif font family
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',       # Use Computer Modern math font
        'figure.dpi': 120,              
        'figure.figsize': (7, 5),       
        'axes.labelsize': 17,           
        'axes.titlesize': 17,           
        'xtick.labelsize': 15,          
        'ytick.labelsize': 15,          
        'legend.fontsize': 15,          
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'font.size': 16,                
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}', # Added bm package
        'mathtext.default': 'regular',   # Use regular (serif) font for math
        'axes.formatter.use_mathtext': True,  # Use mathtext for axis formatting
    })
    # Read the data
    with open(results_file, 'r') as f:
        results = json.load(f)

    # getting mle
    full_dir = os.path.join("mnist_test", "log", "fp+errors")
    results_file = os.path.join(full_dir, f"mle_n_hidden_{n_hidden}_data_{name_data}_feature_{feature_name}.json")
    print('plotting test errors vs alpha for ', results_file)
    if not os.path.exists(results_file):
        print(f"No mle results file found at {results_file}")
        return
    with open(results_file, 'r') as f:
        mle_results = json.load(f)
    # Extract alphas and errors



    for error_type in ["test_error", "train_error", "missclass_test_error"]:
        fig, ax = plt.subplots()
        if error_type == "missclass_test_error":
            mle_errors_type = "mle_misclass_test_errors"
        else:
            mle_errors_type = "mle_"+error_type+"s"

        alphas = []
        mle_alphas = []
        mle_errors = []
        mle_stds = []
        theoretical_errors = []
        color = 'darkblue' if feature_name == "tanh" else 'darkgreen'
        for alpha_str, data in results.items():
            # Skip if this data point doesn't match our parameters
            alpha = float(alpha_str)
            if data.get("n_hidden") != n_hidden or data.get("classes_to_keep") != classes_to_keep:
                continue
            if alpha < alpha_min:
                continue
            if error_type == "test_error" and alpha < 4.9:
                continue
                
            alphas.append(alpha)
            theoretical_errors.append(data[error_type])

        for alpha_str, data in mle_results.items():
            alpha = float(alpha_str)
            if data.get("n_hidden") != n_hidden or data.get("classes_to_keep") != classes_to_keep:
                continue
            if alpha < alpha_min:
                continue
            if error_type == "test_error" and alpha < 5.4:
                continue

            mle_alphas.append(alpha)
            mle_errors.append(np.mean(data[mle_errors_type]))
            mle_stds.append(np.std(data[mle_errors_type]))
        
        # Sort by alpha
        sorted_indices = np.argsort(alphas)
        alphas = np.array(alphas)[sorted_indices]
        theoretical_errors = np.array(theoretical_errors)[sorted_indices]

        sorted_indices_mle = np.argsort(mle_alphas)
        mle_alphas = np.array(mle_alphas)[sorted_indices_mle]
        mle_errors = np.array(mle_errors)[sorted_indices_mle]
        mle_stds = np.array(mle_stds)[sorted_indices_mle]
        
        # Create the plot
        
        # Plot theoretical curve
        ax.plot(alphas,
                theoretical_errors, 
                '-', 
                color=color,
                label='Theoretical prediction', 
                linewidth=2)
        
        # Plot empirical results with error bars
        ax.errorbar(
            x=mle_alphas,
            y=mle_errors,
            yerr=mle_stds/10,  # Standard error
            color='blue',
            fmt='o',  # markers
            markersize=3,
            capsize=2,
            capthick=1,
            elinewidth=1.5,
            alpha=0.5,
            label=r'Empirical (mean $\pm$ s.e.)'  # Added legend
        )
        
        # Add legend
        ax.legend()
        
        # Axis labels, title, legend
        if error_type == "missclass_test_error":
            y_label = "Classification error"
        elif error_type == "train_error":
            y_label = "Train error"
        else:
            y_label = "Test error"
        ax.set_xlabel(r'$\alpha$')
       # ax.set_ylabel(y_label)
        if feature_name == "tanh":
            ax.set_title(f"Random Tanh Features ({n_hidden} units)")
        else:
            ax.set_title(f"Random ReLU Features ({n_hidden} units)")
        ax.legend()

        ax.grid(True)
        plt.tight_layout()


        
        # Save the plot
        save_dir = os.path.join("mnist_test", "log", "figures", error_type)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{error_type}_errors_vs_alpha_n_hidden_{n_hidden}feature_{feature_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"Plot saved to {save_path}")
