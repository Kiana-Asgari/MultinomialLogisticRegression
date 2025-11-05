import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import matplotlib as mpl
from matplotlib.lines import Line2D

from multinomial_logistic.log_data.log_fp_tests import get_fp_statistics
from multinomial_logistic.log_data.read_mle_empirical import read_mle_results
from multinomial_logistic.log_data.log_esd import get_density_data
from multinomial_logistic.log_data.read_mle_empirical import read_mle_results
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from multinomial_logistic.log_data.log_fp_tests import read_fp_tests_regularized
from multinomial_logistic.log_data.read_mle_empirical import read_mle_results
from multinomial_logistic.log_data.read_mle_empirical import get_mle_statistics


###############################################################################


import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def _find_matching_R00_key(R_00_target, keys, precision=3):

    # Convert target to numpy array and round
    R_00_target_arr = np.array(R_00_target)
    R_00_target_rounded = np.round(R_00_target_arr, precision)
    
    for key in keys:
        try:
            # Parse the key as a numpy array
            # Handle both str(array) format and JSON-like format
            key_arr = eval(key.replace('array(', '').replace(')', ''))
            key_arr = np.array(key_arr)
            key_rounded = np.round(key_arr, precision)
            
            # Check if arrays match
            if np.allclose(R_00_target_rounded, key_rounded, atol=0):
                return key
        except:
            # If parsing fails, skip this key
            continue
    
    return None





def plot_regularized_error(k, 
                           k_0, 
                           R_00, 
                           save_path=None, 
                           emp_values = None,
                           emp_window=0., 
                           lambda_reg_max=0.39,
                           lambda_reg_min=0):

    print('[Info] Plotting regularized error...')

    # ---------------------------------
    # 1. Load or retrieve your data
    # ---------------------------------
    results = read_fp_tests_regularized(k, k_0, R_00)
    print('loaded the theoretical results', results.keys())
    emp_results = get_mle_regularized_results(k, k_0, R_00, d=250)
    print('loaded the empirical results', emp_results.keys())

    if results is None:
        print("[Warning] No regularized test results found.")
        #return

    # ------------------------------
    # 2. Create figure directory
    # ------------------------------
    if save_path is None:
        fig_dir = os.path.join(
            os.path.dirname(__file__), 
            "figures", 
            "errors", 
            "regularized"
        )
        os.makedirs(fig_dir, exist_ok=True)

    # --------------------------------------
    # 3. Configure matplotlib & Seaborn RC
    # --------------------------------------
    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7}) 
    mpl.rcParams.update({
        'text.usetex': True,            # For LaTeX rendering
        'font.family': 'serif',         # Use serif font family
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',       # Use Computer Modern math font
        'figure.dpi': 120,              
        'figure.figsize': (7, 5),       
        'axes.labelsize': 18,           
        'axes.titlesize': 18,           
        'xtick.labelsize': 18,          
        'ytick.labelsize': 18,          
        'legend.fontsize': 18,          
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'font.size': 18,                
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}', # Added bm package
        'mathtext.default': 'regular',   # Use regular (serif) font for math
        'axes.formatter.use_mathtext': True,  # Use mathtext for axis formatting
    })
    unique_alphas = sorted(results.keys())
    # Colors for different alpha lines
    colors = [
        'black',
        '#002B5B',  # Darkest navy blue
        '#3E7893',  # Medium blue
        '#5091AA',  # Blue gray
        '#62A9C1',  # Light steel blue
        '#74C2D8',  # Sky blue
        '#86DBEF',  # Light blue
        '#98F4FF'   # Lightest blue
        ]


    # -----------------------------
    # 4. Iterate over "test" & "train"
    # -----------------------------
    for error_type in ['test_errors', 'train_errors', 'misclassification_test_errors', 'norms']:
        fig, ax = plt.subplots()

        for i, alpha in enumerate(unique_alphas):
            # Optionally skip alpha == 2.0 if you want
            if alpha == 2.0:
                continue

            data = results[alpha]

            # Unpack and filter by lambda_reg_max
            lambda_values_all = data['lambda_values'].flatten()
            test_errors_all = data['test_errors'].flatten()
            train_errors_all = data['train_errors'].flatten()
            misclassification_test_errors_all = data['misclassification_test_errors'].flatten()
            f_norms_all = data['f_norms'].flatten()
       


            filtered = [
                (l, t, tr, mte, fn) for (l, t, tr, mte, fn) 
                in zip(lambda_values_all,
                        test_errors_all, 
                        train_errors_all, 
                        misclassification_test_errors_all,
                        f_norms_all) 
                if l < lambda_reg_max and l > lambda_reg_min
            ]
            if not filtered:
                continue

            lambda_values, test_errors, train_errors, misclassification_test_errors, f_norms = zip(*filtered)


            # Decide which error array to plot
            if error_type == 'test_errors':
                errors = test_errors
                y_label = 'Test error (log loss)'

            elif error_type == 'train_errors':
                errors = train_errors
                y_label = 'Train error'

            elif error_type == 'misclassification_test_errors':
                errors = misclassification_test_errors
                y_label = 'Test error (classification)'

            elif error_type == 'norms':
                errors = np.sqrt(f_norms)
                y_label =  f'Estimation error ($\|\\bold{{\Theta}} - \\bold{{\Theta_0}}\\|_F$)'

            # Add empirical points if available
            _plot_empirical_errors(
                    ax=ax,
                    alpha=alpha,
                    emp_results=emp_results,
                    error_type=error_type,
                    emp_window=emp_window,
                    lambda_reg_max=lambda_reg_max,
                    lambda_reg_min=lambda_reg_min,
                    emp_values=emp_values,
                    color=colors[i]
                )


            # Plot theoretical or main curve
            if alpha == 10:
                lambda_values = lambda_values[1:]
                errors = errors[1:]
            ax.plot(
                2*np.array(lambda_values), 
                errors, 
                '-', 
                color=colors[i],
                label=fr'$\alpha={alpha:.1f}$',
                linewidth=1.5
            )


        # Axis labels, title, legend
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(y_label)


        ax.legend()
        ax.grid(True)


        plt.tight_layout()

        # 5. Save Plot
        if save_path is None:
            save_path_final = os.path.join(
                fig_dir, 
                f'reg_{error_type}_error_k{k}_k0{k_0}.pdf'
            )
        else:
            # Modify user-provided save_path to differentiate test vs train
            base, ext = os.path.splitext(save_path)
            save_path_final = f'{base}_{error_type}{ext}'

        plt.savefig(save_path_final, dpi=300, bbox_inches='tight')
        print(f'[Info] {error_type.title()} error plot saved to {save_path_final}')
        plt.close(fig)


def _plot_empirical_errors(ax,
                           alpha,
                           emp_results,
                           error_type,
                           emp_window,
                           lambda_reg_max,
                           lambda_reg_min,
                           emp_values,
                           color):
    
    print('*****ploting emp for alpha:', alpha, 'error_type:', error_type)
    # Retrieve dictionary of {lambda_value: {error_type: [list_of_errors]}}
    emp_lambda_errors = emp_results[alpha]

    # Sort by lambda
    emp_lambdas_sorted = sorted(emp_lambda_errors.keys())
    filtered_indices = []

    last_lambda = -10
    # Filter out points that are too close together or above max
    for j, lambd in enumerate(emp_lambdas_sorted):
        if emp_values is not None:
            if lambd not in emp_values:
                print(f"[Warning] Lambda {lambd} not in emp_values")
                continue
        elif (lambd - last_lambda >= emp_window) and (lambd < lambda_reg_max) and (lambd > lambda_reg_min):
            filtered_indices.append(j)
            last_lambda = lambd

    # Extract mean, std for the chosen error_type
    filtered_lambdas = [emp_lambdas_sorted[i] for i in filtered_indices]
    if error_type == 'norms':   
        filtered_errors = [
            np.mean(np.sqrt(emp_lambda_errors[lambd][error_type])) for lambd in filtered_lambdas
        ]
        filtered_stds = [
            np.std(np.sqrt(emp_lambda_errors[lambd][error_type]), ddof=1) 
            for lambd in filtered_lambdas
        ]
    else:
        filtered_errors = [
            np.mean(emp_lambda_errors[lambd][error_type]) for lambd in filtered_lambdas
        ]
        filtered_stds = [
            np.std(emp_lambda_errors[lambd][error_type], ddof=1) 
            for lambd in filtered_lambdas
        ]

    print('filtered_lambdas:', filtered_lambdas)
    print('filtered_errors:', filtered_errors)
    ax.errorbar(
        x=2*np.array(filtered_lambdas),
        y=filtered_errors,
        yerr=np.array(filtered_stds)/10,
        color=color,
        fmt='o',  # square markers
        markersize=3,
        capsize=2.5,
        capthick=1,
        elinewidth=1.5,
        alpha=0.7
    )


#################################################

def set_up_plotting_style():
            # Create figure with custom style
    sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
                               'axes.linewidth': 0.7})
    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'legend.fontsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'font.size': 14,
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}',
        'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True,
    })
    colors = [
        '#E57A77',
        '#1F449C',  # Deep ocean blue
        'mediumslateblue',  # Medium blue
        '#5091AA',  # Blue gray
        '#62A9C1',  # Light steel blue
        '#74C2D8',  # Sky blue
        '#86DBEF',  # Light blue
        '#98F4FF'   # Lightest blue
        ]
    return colors





from multinomial_logistic.log_data.log_fp_tests import get_fp_misclassification_statistics
from configs.R_initiation import get_R_00

def plot_errors_vs_alpha(k, k_0, alpha_max, alpha_min):
    colors = set_up_plotting_style()
    if k>=3:
        fp_results_symmetric = get_fp_statistics(k, k_0, type_3='symmetric')
        fp_results_non_symmetric= get_fp_statistics(k, k_0, type_3='three_classes_close')
        fp_results_two_classes_close=get_fp_statistics(k, k_0, type_3='two_classes_close')
        fp_results_two_vs_two_vs_one=get_fp_statistics(k, k_0, type_3='two_vs_two_vs_one')
    else:
        fp_results_two_classes_close = get_fp_statistics(k, k_0, two_classes_close=True)
        fp_results_symmetric = get_fp_statistics(k, k_0, non_symmetric=False, two_classes_close=False)
        fp_results_non_symmetric = get_fp_statistics(k, k_0, two_classes_close=False, non_symmetric=True)
        fp_results_two_vs_two_vs_one = None
    if fp_results_symmetric is None or fp_results_two_classes_close is None or fp_results_non_symmetric is None:
        print("No FP results found")
        return
    
    alphas_fp_symmetric, test_errors_fp_symmetric, train_errors_fp_symmetric, F_norms_fp_symmetric, misclassification_test_errors_fp_symmetric = fp_results_symmetric
    alphas_fp_two_classes_close, test_errors_fp_two_classes_close, train_errors_fp_two_classes_close, F_norms_fp_two_classes_close, misclassification_test_errors_fp_two_classes_close = fp_results_two_classes_close
    alphas_fp_non_symmetric, test_errors_fp_non_symmetric, train_errors_fp_non_symmetric, F_norms_fp_non_symmetric, misclassification_test_errors_fp_non_symmetric = fp_results_non_symmetric
    alphas_fp_two_vs_two_vs_one, test_errors_fp_two_vs_two_vs_one, train_errors_fp_two_vs_two_vs_one, F_norms_fp_two_vs_two_vs_one, misclassification_test_errors_fp_two_vs_two_vs_one = fp_results_two_vs_two_vs_one
    

    # Read MLE data
    if k>3:
        base_path = 'multinomial_logistic/log_data/Oct_data/mle_empirical'
        d = 300
    elif k==3:
        base_path = 'multinomial_logistic/log_data/Oct_data/mle_empirical'
        d = 250
    else:
        base_path = 'multinomial_logistic/log_data/newdata/mle_empirical'
        d = 250

    # Load mle data
    if k>=3:
        with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_symmetric.json'), 'r') as f:
            mle_data_symmetric = json.load(f)
        with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_two_classes_close.json'), 'r') as f:
            mle_data_two_classes_close = json.load(f)
        with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_three_classes_close.json'), 'r') as f:
            mle_data_three_classes_close = json.load(f)
        with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_two_vs_two_vs_one.json'), 'r') as f:
            mle_data_two_vs_two_vs_one = json.load(f)

    else:
        with open(os.path.join(base_path, f'mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150.json'), 'r') as f:
            mle_data_symmetric = json.load(f)
        with open(os.path.join(base_path, f'mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150_two_classes_close.json'), 'r') as f:
            mle_data_two_classes_close = json.load(f)
        with open(os.path.join(base_path, f'mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150_non_symmetric.json'), 'r') as f:
            mle_data_non_symmetric = json.load(f)
        with open(os.path.join(base_path, f'mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150_two_vs_two_vs_one.json'), 'r') as f:
            mle_data_two_vs_two_vs_one = json.load(f)

    # Plot each metric
    metrics = [
        ('test_errors', [test_errors_fp_symmetric, test_errors_fp_two_classes_close, test_errors_fp_non_symmetric, test_errors_fp_two_vs_two_vs_one],
         'Test error (log loss)'),
        ('train_errors', [train_errors_fp_symmetric, train_errors_fp_two_classes_close, train_errors_fp_non_symmetric, train_errors_fp_two_vs_two_vs_one],
         'Train error'),
        ('misclassification_test_errors', [misclassification_test_errors_fp_symmetric, misclassification_test_errors_fp_two_classes_close, misclassification_test_errors_fp_non_symmetric, misclassification_test_errors_fp_two_vs_two_vs_one],
         'Test error (classification)'),
        ('F_norm', [F_norms_fp_symmetric, F_norms_fp_two_classes_close, F_norms_fp_non_symmetric, F_norms_fp_two_vs_two_vs_one],
         f'Estimation error ($\|\\bold{{\Theta}} - \\bold{{\Theta_0}}\\|_F$)')
    ]

    for metric_name, fp_values, y_label in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot theoretical curves
        labels = [r'$\mathbf{R}_{00}= sym \mathbf{R}_{00}^{(1)}$', 
                 r'$\mathbf{R}_{00}= two \mathbf{R}_{00}^{(2)}$', 
                 r'$\mathbf{R}_{00}= three \mathbf{R}_{00}^{(3)}$',
                 r'$\mathbf{R}_{00}= two \mathbf{R}_{00}^{(2)}$']
        
        alphas_list = [alphas_fp_symmetric, alphas_fp_two_classes_close, alphas_fp_non_symmetric, alphas_fp_two_vs_two_vs_one]
        
        for i, (alphas, values, label, color) in enumerate(zip(alphas_list, fp_values, labels, colors)):
            # Plot each metric separately for better debugging
            if metric_name == 'test_errors':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max)
                filtered_alphas = alphas[mask]
                filtered_values = np.array(values[mask]) #+ 2*1e-3
            elif metric_name == 'train_errors':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max)
                filtered_alphas = alphas[mask]
                filtered_values = np.array(values[mask])
            elif metric_name == 'misclassification_test_errors':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max)
                filtered_alphas = alphas[mask]
                filtered_values = np.array(values[mask])
            elif metric_name == 'F_norm':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max)
                filtered_alphas = alphas[mask]
                filtered_values = np.sqrt(np.array(values[mask]))# + 1e-2
            plt.plot(filtered_alphas, filtered_values, '-', color=colors[2-i], label=label, linewidth=1.8)

        # Add empirical points 
        if k>=3:
            data_sets = [
                (mle_data_symmetric, get_R_00(k, 'symmetric'), colors[2]),
                (mle_data_two_classes_close, get_R_00(k, 'two_classes_close'), colors[1]),
                (mle_data_three_classes_close, get_R_00(k, 'three_classes_close'), colors[0]),
                (mle_data_two_vs_two_vs_one, get_R_00(k, 'two_vs_two_vs_one'), colors[3])
            ]
        else:
            data_sets = [
                (mle_data_symmetric, [[1.0, 0.5], [0.5, 1.0]], colors[2]),
                (mle_data_two_classes_close, [[1.0, 0.9], [0.9, 1.0]], colors[1]),
                (mle_data_non_symmetric, [[1.0, -0.5], [-0.5, 1.0]], colors[0])
            ]

        
        for data, R_00, color in data_sets:
            # Find matching key with 3-digit precision
            matching_key = _find_matching_R00_key(R_00, data['results'].keys())
            
            if matching_key is not None:
                alphas, values, errors = [], [], [] 

                for alpha_str, result in data['results'][matching_key].items():
                    alpha = float(alpha_str)

                    if alpha_min <= alpha <= alpha_max:
                        mle_metric_name = 'norm' if metric_name == 'F_norm' else metric_name                                
                        metric_values = result[mle_metric_name]                                
                        if not isinstance(metric_values, (list, np.ndarray)):
                            metric_values = [metric_values]
                        
                        alphas.append(alpha)
                        values.append(np.mean(metric_values))
                        errors.append(2*np.std(metric_values) / (np.sqrt(len(metric_values))))
                
                if alphas:
                    # Debug prints
                    print(f"Final mle lengths for {metric_name} - alphas: {len(alphas)}, values: {len(values)}, errors: {len(errors)}")
                    print('alphas:', alphas)
                    alphas = np.array(alphas)
                    values = np.array(values)
                    errors = np.array(errors)
                    
                    # Sort by alpha to ensure proper plotting
                    sort_idx = np.argsort(alphas)
                    alphas = alphas[sort_idx]
                    values = values[sort_idx]
                    errors = errors[sort_idx]
                    if metric_name == 'test_errors' or metric_name == 'F_norm':
                        alphas = np.array(alphas) - 2*1e-3
                       #plt.ylim(bottom=1)

                    #if metric_name == 'misclassification_test_errors':
                    #    plt.ylim(bottom=0.4,top=0.62)
                    plt.errorbar(alphas, values, yerr=errors, fmt='o', 
                                    color=color,        
                                    markersize=3,
                                    capsize=2.8,
                                    capthick=1.5,
                                    elinewidth=1.8,
                                    alpha=0.76)

        plt.xlabel(r'$\alpha$')
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.legend()
        #plt.xlim(left=5)
        #plt.ylim(top=2)
        
        # Add formatter for y-axis ticks to show 2 decimal places
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        # Save the plot
        save_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "figures", f"{k}_classes", "errors_vs_alpha")
        os.makedirs(save_dir, exist_ok=True)
        saving_path = os.path.join(save_dir, f'{metric_name}_vs_alpha_k{k}_k0{k_0}.pdf')
        plt.savefig(saving_path, bbox_inches='tight', dpi=300)
        print(f'\n{metric_name} vs alpha plot saved to {saving_path}')
        plt.close()




















def plot_density(k,
                 k_0, 
                 R_00, 
                 alpha_target, 
                 eigenvalues=None, 
                 density_lower_bound=0, 
                 d=250, 
                 lambda_reg=0,
                 save_path=None, 
                 emp_histogram=False, 
                 emp_bins=80, 
                 z_real_lower_bound=0.0, 
                 clean_data_for_5=False, 
                 clean_data_for_3=False, 
                 clean_data_for_10=False,
                 clean_data_for_20=False):
    print('*****plotting density for alpha:', alpha_target, 'n_bins:', emp_bins)


    # ---------------------------
    # 1. Configure Figure & Fonts
    # ---------------------------
    # You can adjust values to fit your own style or conference guidelines.
    # If you want LaTeX rendering, uncomment text.usetex lines and ensure
    # LaTeX is installed on your system.
    
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

    # -------------------------------------
    # 2. Retrieve and filter theoretical data
    # -------------------------------------

    closest_alpha, z_reals, z_imags, densities = get_density_data(
        k, 
        k_0, 
        R_00, 
        alpha_target, 
        lambda_reg,
        clean_data_for_5=clean_data_for_5,
        clean_data_for_3=clean_data_for_3
    )
    
    if closest_alpha is None:
        print("[Warning] No theoretical data found for the given parameters.")
        return

    # Filter out densities below the specified lower bound
    mask = densities > density_lower_bound
    z_reals_filtered = z_reals[mask]
    densities_filtered = densities[mask]

    # Further filter: keep only z_reals >= z_real_lower_bound
    mask_z_real = z_reals_filtered > z_real_lower_bound
    z_reals_filtered = z_reals_filtered[mask_z_real]
    densities_filtered = densities_filtered[mask_z_real]

    for i in range(len(z_reals_filtered)):
        print(f'[index {i}]: z:', z_reals_filtered[i], 'density:', densities_filtered[i])

    fig, ax = plt.subplots()

    # Plot the theoretical density curve
    ax.plot(
        z_reals_filtered, 
        densities_filtered, 
        color='darkblue', 
        linewidth=2.4, 
        label=r'$\mu_{\star}(\nu^{\mathrm{opt}})$'
    )

    if emp_histogram and eigenvalues is not None:
        if isinstance(eigenvalues, np.ndarray):
            flat_eigenvalues = eigenvalues.flatten()
        else:
            flat_eigenvalues = np.array(eigenvalues).flatten()
        
        sns.histplot(
            data=flat_eigenvalues,
            bins=emp_bins,
            stat='density',
            fill=True,
            alpha=0.5,
            #linewidth=1,
            color='blue',
            label='Empirical spectrum',
            ax=ax
        )

    # -----------------
    # 5. Final touches
    # -----------------
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    ax.set_xlabel(r'$\lambda$')  # If you want text label: ax.set_xlabel('Lambda')
    ax.set_ylabel('')  # Explicitly set y-label to empty string


    # -------------
    # 6. Save Plot
    # -------------
    if save_path is None:
        # Create a default path if none is provided
        fig_dir = os.path.join(os.path.dirname(__file__), "Oct_data", "figures", "ESD", f'{k+1}_classes')
        os.makedirs(fig_dir, exist_ok=True)
        filename = (f"density_k{k}_k0{k_0}_R00{R_00[0,1]:.2f}"
                    f"_alpha{alpha_target:.2f}_lambda{lambda_reg}"
                    f"_emp{int(emp_histogram)}_d{d}.pdf")
        save_path = os.path.join(fig_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Info] Plot saved to {save_path}")

    # Close the figure to free up memory
    plt.close(fig)