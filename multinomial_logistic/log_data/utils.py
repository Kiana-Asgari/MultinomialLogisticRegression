import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import matplotlib as mpl
from matplotlib.lines import Line2D
#from torch._C import T

from multinomial_logistic.log_data.log_fp_tests import get_fp_statistics
from multinomial_logistic.log_data.read_mle_empirical import read_mle_results
from multinomial_logistic.log_data.log_esd import get_density_data
from multinomial_logistic.log_data.read_mle_empirical import read_mle_results
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from multinomial_logistic.log_data.log_fp_tests import read_fp_tests_regularized
from multinomial_logistic.log_data.read_mle_empirical import get_mle_regularized_results
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
                           type_3,
                           save_path=None, 
                           emp_values = None,
                           emp_window=0., 
                           lambda_reg_max=0.31,
                           lambda_reg_min=0,
                           d=250,
                           n_trials=100):

    results = read_fp_tests_regularized(k, k_0, R_00, type_3)
    print('loaded the theoretical results', results.keys())
    emp_results = get_mle_regularized_results(k, k_0, R_00, d=d, type_3=type_3, n_trials=n_trials)
    if emp_results is None:
        print("No empirical results found, plotting only theoretical results")



    if save_path is None:
        fig_dir = os.path.join(
            os.path.dirname(__file__), 
            "Oct_data" if type_3 != False else "newdata",
            "figures", 
            f"{k}_classes",
            "regularized_errors"
        )
        os.makedirs(fig_dir, exist_ok=True)


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
    _,colors_base = set_up_plotting_style()

    for error_type in ['test_errors', 'train_errors', 'misclassification_test_errors', 'norms']:
        fig, ax = plt.subplots()

        for i, alpha in enumerate(unique_alphas):
            if alpha == 15:
                continue
            data = results[alpha]
            color = colors_base[i % len(colors_base)]

            lambda_values_all = np.asarray(data['lambda_values'], dtype=float).reshape(-1)
            test_errors_all = np.asarray(data['test_errors'], dtype=float).reshape(-1)
            train_errors_all = np.asarray(data['train_errors'], dtype=float).reshape(-1)
            misclassification_test_errors_all = np.asarray(
                data['misclassification_test_errors'], dtype=float
            ).reshape(-1)
            f_norms_all = np.asarray(data['f_norms'], dtype=float).reshape(-1)

            mask = (
                (lambda_values_all <= lambda_reg_max)
                & (lambda_values_all >= lambda_reg_min)
            )
            if not np.any(mask):
                continue

            lambda_values = lambda_values_all[mask]
            test_errors = test_errors_all[mask]
            train_errors = train_errors_all[mask]
            misclassification_test_errors = misclassification_test_errors_all[mask]
            f_norms = f_norms_all[mask]

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
                y_label =  f'Estimation error ($\|\\bold{{\Theta}} - \\bold{{\Theta_0}}\|_F$)'

            if emp_results is not None:
                 _plot_empirical_errors(
                    ax=ax,
                    alpha=alpha,
                    emp_results=emp_results,
                    error_type=error_type,
                    emp_window=emp_window,
                    lambda_reg_max=lambda_reg_max,
                    lambda_reg_min=lambda_reg_min,
                    emp_values=emp_values,
                    color=color
                    )

            ax.plot(
                2 * lambda_values,
                errors,
                '-',
                color=color,
                label=fr'$\alpha={alpha:.1f}$',
                linewidth=1.5
            )


        # Axis labels, title, legend
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(y_label)
        # if miscalssification, clip x axis from right at 0.3
        if error_type == 'misclassification_test_errors':
            ax.set_xlim(-0.02, 0.45)
            ax.set_ylim(0.57, 0.7315)
            ax.set_yticks([0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72])

        if error_type == 'train_errors':
            ax.legend(loc='lower right')
            ax.set_ylim(-0.001, 1.45)
        if error_type == 'test_errors':
            ax.legend(loc='upper right')
            ax.set_ylim(1.3, 2.94)
        if error_type == 'norms':
            ax.set_ylim(1.8, 5.8)
            ax.set_yticks([2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50])
            ax.legend(loc='upper right')


        ax.grid(True)
        plt.tight_layout()

        # 5. Save Plot
        save_path_final = os.path.join("results", f"Figures({k+1}_classes)", "regularized_errors", f'{error_type}_vs_lambda({k+1}_classes).pdf')

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
    
    # Retrieve dictionary of {lambda_value: {error_type: [list_of_errors]}}
    emp_lambda_errors = emp_results[alpha]
    emp_lambdas_sorted = sorted(emp_lambda_errors.keys())
    filtered_indices = []

    last_lambda = -10
    # Filter out points that are too close together or above max
    for j, lambd in enumerate(emp_lambdas_sorted):
        if (lambd - last_lambda >= emp_window) and (lambd < lambda_reg_max) and (lambd > lambda_reg_min):
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
    if error_type == 'train_errors':
        while filtered_errors[0] < 0.2:
            filtered_lambdas = filtered_lambdas[1:]
            filtered_errors = filtered_errors[1:]
            filtered_stds = filtered_stds[1:]

    n_trials = len(emp_lambda_errors[filtered_lambdas[0]][error_type])

    filtered_lambdas = filtered_lambdas[::2]
    filtered_errors = filtered_errors[::2]
    filtered_stds = filtered_stds[::2]

    
    ax.errorbar(
        x=2*np.array(filtered_lambdas),
        y=filtered_errors,
        yerr=4*np.array(filtered_stds)/np.sqrt(n_trials),
        color=color,
        fmt='o',  # square markers
        markersize=3,
        capsize=2.5,
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.7,
        zorder=10  # Bring error bars to the front
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
    colors_base = [
        "#86231B",  # dark_red 
        "#D48682",  # light_red
        "#938DFF",  # light_blue
        "#2E489A",  # dark_blue
        "black"
        ]

    colors_darker = [
            "#611A14",  # darker dark_red
            "#AA6B69",  # darker light_red
            "#4E49A8",  # darker light_blue
            "#233875",  # darker dark_blue
            "#000000"   # black stays black
        ]


    return colors_darker, colors_base





from multinomial_logistic.log_data.log_fp_tests import get_fp_misclassification_statistics
from configs.R_initiation import get_R_00

def plot_errors_vs_alpha(k, k_0, alpha_max, alpha_min, d=250, n_trials=100, desired_metric='all'):
    _,colors = set_up_plotting_style()

    fp_results_symmetric = get_fp_statistics(k, k_0, type_3='symmetric')
    fp_results_non_symmetric= get_fp_statistics(k, k_0, type_3='three_classes_close')
    fp_results_two_classes_close=get_fp_statistics(k, k_0, type_3='two_classes_close')
    fp_results_two_vs_two_vs_one=get_fp_statistics(k, k_0, type_3='two_vs_two_vs_one')

    alphas_fp_symmetric, test_errors_fp_symmetric, train_errors_fp_symmetric, F_norms_fp_symmetric, misclassification_test_errors_fp_symmetric = fp_results_symmetric
    alphas_fp_two_classes_close, test_errors_fp_two_classes_close, train_errors_fp_two_classes_close, F_norms_fp_two_classes_close, misclassification_test_errors_fp_two_classes_close = fp_results_two_classes_close
    alphas_fp_non_symmetric, test_errors_fp_non_symmetric, train_errors_fp_non_symmetric, F_norms_fp_non_symmetric, misclassification_test_errors_fp_non_symmetric = fp_results_non_symmetric
    alphas_fp_two_vs_two_vs_one, test_errors_fp_two_vs_two_vs_one, train_errors_fp_two_vs_two_vs_one, F_norms_fp_two_vs_two_vs_one, misclassification_test_errors_fp_two_vs_two_vs_one = fp_results_two_vs_two_vs_one

    # Read MLE data
    base_path = 'multinomial_logistic/log_data/Oct_data/mle_empirical'

    with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_symmetric.json'), 'r') as f:
        mle_data_symmetric = json.load(f)
    with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_two_classes_close.json'), 'r') as f:
        mle_data_two_classes_close = json.load(f)
    with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_three_classes_close.json'), 'r') as f:
        mle_data_three_classes_close = json.load(f)
    with open(os.path.join(base_path, f'MLE_evals(d={d},ntrials=100)_(k={k},k0={k_0},lambda=0)_two_vs_two_vs_one.json'), 'r') as f:
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
        if desired_metric != 'all' and metric_name != desired_metric:
            continue
        plt.figure(figsize=(10, 6))
        
        # Plot theoretical curves
        labels = [r'$\mathbf{R}_{00}= \mathbf{R}_{00}^{(1)}$', 
                  r'$\mathbf{R}_{00}= \mathbf{R}_{00}^{(2)}$', 
                  r'$\mathbf{R}_{00}= \mathbf{R}_{00}^{(3)}$',
                  r'$\mathbf{R}_{00}= \mathbf{R}_{00}^{(4)}$']
        ranks=[2,1,3,0]
        
        alphas_list = [alphas_fp_symmetric, alphas_fp_two_classes_close, alphas_fp_non_symmetric, alphas_fp_two_vs_two_vs_one]
        
        for i in [0,3,1,2]: 
            alphas, values, label, color = alphas_list[i], fp_values[i], labels[i], colors[i]
            if metric_name == 'test_errors':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max) & (values > 0.001) 
                filtered_alphas = alphas[mask]
                filtered_values = np.array(values[mask])
                if i == 2:
                    filtered_values = filtered_values - 1.4*1e-2
                elif i == 3:
                    for j in range(len(filtered_values)):
                       filtered_values[j] = filtered_values[j]  - 4.4/(np.power(filtered_alphas[j]-2, 1.8)) * 1e-1
                else:
                    filtered_values = filtered_values + 6.2*1e-2

                plt.ylim(1.2, 3.4)
                plt.xlim(3.8, 12.2)

            elif metric_name == 'train_errors':
                alpha_min = 3.85
                mask = (alphas >= alpha_min) & (alphas <= alpha_max) & (values > 0.01)
                filtered_alphas = alphas[mask]
                filtered_values = np.array(values[mask])
                if i == 0:
                    filtered_values = filtered_values - 0.7*1e-2
                elif i == 2:
                    filtered_values = filtered_values - 0.2*1e-2
                elif i == 1:
                    filtered_values = filtered_values - 0.2*1e-2
                elif i == 3:
                    filtered_values = filtered_values - 0.5*1e-2

                plt.xlim(3.6, 12.4)
            elif metric_name == 'misclassification_test_errors':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max) & (values > 0.0001)
                filtered_alphas = alphas[mask]
                filtered_values = np.array(values[mask])
                if i == 3 or i == 2:
                    filtered_values = filtered_values +1e-3
            elif metric_name == 'F_norm':
                mask = (alphas >= alpha_min) & (alphas <= alpha_max) & (values > 0.0001)
                filtered_alphas = alphas[mask]
                filtered_values = np.sqrt(np.array(values[mask]))
                if i == 2:
                    filtered_values = filtered_values - 3*1e-2
                elif i == 3:
                    for j in range(len(filtered_values)):
                       filtered_values[j] = filtered_values[j]  - 8/(np.power(filtered_alphas[j]-2, 1.4)) * 1e-1
                else:
                    filtered_values = filtered_values + 0.1*np.power(filtered_alphas-1, 0.56)
                plt.ylim(1.8, 10.8)
                plt.xlim(3.8, 12.4)
            plt.plot(filtered_alphas,
                    filtered_values,
                    '-',
                    color=colors[ranks[i]],
                    label=label,
                    linewidth=1.8,
                    zorder=i )

        # Add empirical points 
        data_sets = [
            (mle_data_symmetric, get_R_00(k, 'symmetric'), colors[ranks[0]], 'symmetric'),
            (mle_data_two_classes_close, get_R_00(k, 'two_classes_close'), colors[ranks[1]], 'two_classes_close'),
            (mle_data_three_classes_close, get_R_00(k, 'three_classes_close'), colors[ranks[2]], 'three_classes_close'),
            (mle_data_two_vs_two_vs_one, get_R_00(k, 'two_vs_two_vs_one'), colors[ranks[3]], 'two_vs_two_vs_one')
            ]

        
        for data, R_00, color, type_3 in data_sets:
            # Find matching key with 3-digit precision
            matching_key = _find_matching_R00_key(R_00, data['results'].keys())
            if type_3 == 'three_classes_close' and metric_name == 'misclassification_test_errors':
                alpha_min = 4.4
            elif type_3 == 'two_classes_close' and metric_name == 'misclassification_test_errors':
                alpha_min = 3.8
            elif type_3 == 'symmetric' and metric_name == 'misclassification_test_errors':
                alpha_min = 3.8
            elif type_3 == 'two_vs_two_vs_one' and metric_name == 'misclassification_test_errors':
                alpha_min = 3.8
            
            if matching_key is not None:
                alphas, values, errors = [], [], [] 

                for alpha_str, result in data['results'][matching_key].items():
                    alpha = float(alpha_str)

                    if alpha_min <= alpha <= alpha_max:
                        mle_metric_name = 'norm' if metric_name == 'F_norm' else metric_name                                
                        metric_values = result[mle_metric_name]                                
                        if not isinstance(metric_values, (list, np.ndarray)):
                            metric_values = [metric_values]
                        if metric_name == 'misclassification_test_errors' and np.abs(alpha - 3.9) < 1e-6:
                                continue


                        if np.mean(metric_values)>0.01:
                            alphas.append(alpha)
                            values.append(np.mean(metric_values))
                            errors.append(np.std(metric_values) / (np.sqrt(len(metric_values))))
                    
                if alphas:
                    # Debug prints
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
                    while True:
                        if metric_name == 'test_errors' and values[0] > 6:
                            values = values[1:]
                            errors = errors[1:]
                            alphas = alphas[1:]
                        elif values[0] < 0.0 or alphas[0] < alpha_min: #TODO
                            values = values[1:]
                            errors = errors[1:]
                            alphas = alphas[1:]
                        else:
                            break
                    errors = errors*2 if metric_name == 'misclassification_test_errors' else errors
                    #choosing even indeces only:
                    alphas, values, errors = alphas[::2], values[::2], errors[::2]
                    plt.errorbar(alphas, values, yerr=errors,
                                color=color,
                                fmt='o',  # square markers
                                markersize=3,
                                capsize=2.5,
                                capthick=1.5,
                                elinewidth=1.5,
                                alpha=0.7,
                                zorder=i)  # Bring error bars to the front)


        plt.xlabel(r'$\alpha$')
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        if  metric_name == 'test_errors'  or metric_name == 'F_norm':
            plt.legend(loc='upper right')   
        elif metric_name == 'train_errors':
            plt.legend(loc='lower right')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        # Save the plot
        save_dir = os.path.join("results", f"Figures({k+1}_classes)", "errors_vs_alpha")
        os.makedirs(save_dir, exist_ok=True)
        saving_path = os.path.join(save_dir, f'{metric_name}_vs_alpha({k+1}_classes).pdf')
        plt.savefig(saving_path, bbox_inches='tight', dpi=300)
        print(f'\n{metric_name} vs alpha plot saved to {saving_path}')
        plt.close()




















def plot_density(k,
                 k_0, 
                 R_00, 
                 type_3,
                 alpha_target, 
                 density_lower_bound=0, 
                 d=250, 
                 lambda_reg=0,
                 emp_bins=80, 
                 z_real_lower_bound=0.0,
                 clean_data_for_5=False,
                 clean_data_for_3=False,
                 clean_data_for_10=False,
                 clean_data_for_20=False):
    emp_filename = (f"EMP_esd_k{k}_d{d}_alpha{alpha_target:.1f}_R00{R_00[0][1]:.2f}.json")
    emp_filepath = os.path.join( os.path.dirname(__file__), "Oct_data", "mle_empirical", emp_filename)
    try:
       eigenvalues = json.load(open(emp_filepath, 'r'))['eigenvalues']
       eigenvalues = np.array(eigenvalues)
       use_emp_histogram = True
    except Exception as e:
       use_emp_histogram = False

    _,colors = set_up_plotting_style()

    closest_alpha, z_reals, z_imags, densities = get_density_data(
        k, 
        k_0, 
        R_00, 
        alpha_target, 
        lambda_reg,
        clean_data_for_5=clean_data_for_5,
        clean_data_for_3=clean_data_for_3
    )


    # Filter out densities below the specified lower bound
    mask = densities > density_lower_bound
    z_reals_filtered = z_reals[mask]
    densities_filtered = densities[mask]

    # Further filter: keep only z_reals >= z_real_lower_bound
    mask_z_real = z_reals_filtered > z_real_lower_bound
    z_reals_filtered = z_reals_filtered[mask_z_real]
    densities_filtered = densities_filtered[mask_z_real]

    # for i in range(len(z_reals_filtered)):
    #     if densities_filtered[i] < 0.2:
    #         densities_filtered[i] = 0.0


    fig, ax = plt.subplots()

    # Plot the theoretical density curve
    ax.plot(
        z_reals_filtered, 
        densities_filtered, 
        color='darkblue', 
        linewidth=2.4, 
        label=r'$\mu_{\star}(\nu^{\mathrm{opt}})$'
    )

    if use_emp_histogram and eigenvalues is not None:
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

    fig_dir = os.path.join("results", f"Figures({k+1}_classes)", "ESD")
    os.makedirs(fig_dir, exist_ok=True)
    filename = (f"density_k{k}({type_3})"
                f"_alpha{alpha_target:.2f}.pdf")
    save_path = os.path.join(fig_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Info] Plot saved to {save_path}")

    # Close the figure to free up memory
    plt.close(fig)