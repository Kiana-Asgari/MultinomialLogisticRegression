import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import matplotlib as mpl
from matplotlib.lines import Line2D

from multinomial_logistic.log_data.log_fp_tests import get_fp_statistics
from multinomial_logistic.log_data.log_mle_empirical import get_mle_statistics
from multinomial_logistic.log_data.log_esd import get_density_data
from multinomial_logistic.log_data.log_mle_empirical import read_emp_esd_results
from multinomial_logistic.evaluation.log_loss_test_error import test_error
from multinomial_logistic.evaluation.log_loss_train_eror import train_error
from multinomial_logistic.log_data.log_fp_tests import read_fp_tests_regularized
from multinomial_logistic.log_data.log_mle_empirical import get_mle_regularized_results



###############################################################################


import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



def plot_regularized_error_vs_d(k, 
                           k_0, 
                           R_00, 
                           save_path=None, 
                           emp_values = None,
                           emp_window=0, 
                           lambda_reg_max=0.6):


    print('[Info] Plotting regularized error...')

    # ---------------------------------
    # 1. Load or retrieve your data
    # ---------------------------------
    results = read_fp_tests_regularized(k, k_0, R_00)
    
    emp_results_20 = get_mle_regularized_results(k, k_0, R_00, d=20)
    emp_results_50 = get_mle_regularized_results(k, k_0, R_00, d=50)
    emp_results_100 = get_mle_regularized_results(k, k_0, R_00, d=100)
    emp_results_250 = get_mle_regularized_results(k, k_0, R_00, d=250)
    

    if results is None:
        print("[Warning] No regularized test results found.")
        return

    # ------------------------------
    # 2. Create figure directory
    # ------------------------------
    if save_path is None:
        fig_dir = os.path.join(
            os.path.dirname(__file__), 
            "figures", 
            "errors", 
            "regularized_vs_d"
        )
        os.makedirs(fig_dir, exist_ok=True)

    # --------------------------------------
    # 3. Configure matplotlib & Seaborn RC
    # --------------------------------------
    sns.set_theme(context='paper', style='whitegrid')
    #sns.set_style("whitegrid", {'axes.edgecolor': 'darkgray',
    #                           'axes.linewidth': 0.6}) 
    mpl.rcParams.update({
        # 'text.usetex': True,     # Uncomment if LaTeX is installed and desired
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],  # Specific serif font
        'mathtext.fontset': 'cm',  # For consistent math font
        'figure.dpi': 120,
        'figure.figsize': (7, 5),
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2
    })

    unique_alphas = sorted(results.keys())
    # Colors for different alpha lines
    colors = [
        'black',
        'pink',
        '#002B5B',  # Darkest navy blue
        '#1B4965',  # Deep ocean blue
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
    for error_type in ['test', 'train']:
        fig, ax = plt.subplots()

        for i, alpha in enumerate(unique_alphas):
            # Optionally skip alpha == 2.0 if you want
            if alpha != 3.0:
                continue

            print(f'[Info] alpha={alpha}, error_type={error_type}')

            data = results[alpha]
            if isinstance(data, tuple) and len(data) == 3:
                # Unpack and filter by lambda_reg_max
                lambda_values_all, test_errors_all, train_errors_all = data
                filtered = [
                    (l, t, tr) for (l, t, tr) 
                    in zip(lambda_values_all, test_errors_all, train_errors_all) 
                    if l < lambda_reg_max
                ]
                if not filtered:
                    continue

                lambda_values, test_errors, train_errors = zip(*filtered)

                # Decide which error array to plot
                if error_type == 'test':
                    errors = test_errors
                    # Add empirical points if available
                    if emp_results_20 and alpha in emp_results_20:
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_20,
                            error_type='test_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color='red'
                        )
                    if emp_results_50 and alpha in emp_results_50:
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_50,
                            error_type='test_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color='darkred'
                        )
                    if emp_results_100 and alpha in emp_results_100:
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_100,
                            error_type='test_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color='black'
                        )
                    if emp_results_250 and alpha in emp_results_250:
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_250,
                            error_type='test_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color='green'
                        )
                else:  # error_type == 'train'
                    errors = train_errors
                    # Add empirical points if available
                    if emp_results_20 and alpha in emp_results_20:
                        print(f"[Info] Plotting empirical errors for alpha={alpha}, error_type=train_errors, emp_results=emp_results_20")
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_20,
                            error_type='train_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color=colors[i]
                        )
                    if emp_results_50 and alpha in emp_results_50:
                        print(f"[Info] Plotting empirical errors for alpha={alpha}, error_type=train_errors, emp_results=emp_results_50")
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_50,
                            error_type='train_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color=colors[i]
                        )
                    if emp_results_100 and alpha in emp_results_100:
                        print(f"[Info] Plotting empirical errors for alpha={alpha}, error_type=train_errors, emp_results=emp_results_100")
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_100,
                            error_type='train_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color=colors[i]
                        )
                    if emp_results_250 and alpha in emp_results_250:
                        _plot_empirical_errors(
                            ax=ax,
                            alpha=alpha,
                            emp_results=emp_results_250,
                            error_type='train_errors',
                            emp_window=emp_window,
                            lambda_reg_max=lambda_reg_max,
                            emp_values=emp_values,
                            color='green'
                        )   

                # Plot theoretical or main curve

                ax.plot(
                    2*np.array(lambda_values), 
                    errors, 
                    '-', 
                    color='blue',
                    label=fr'$\alpha={alpha:.1f}$',
                    linewidth=1.5
                )
            else:
                print(f"[Warning] Unexpected data format for alpha={alpha}. Skipping.")

        # Axis labels, title, legend
        ax.set_xlabel(r'$\lambda$ (regularization)')
        ax.set_ylabel(f'{error_type.title()} error')


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






def plot_regularized_error(k, 
                           k_0, 
                           R_00, 
                           save_path=None, 
                           emp_values = None,
                           emp_window=0.4, 
                           lambda_reg_max=0.6,
                           lambda_reg_min=0.3):

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
        'pink',
        '#002B5B',  # Darkest navy blue
        '#1B4965',  # Deep ocean blue
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
                y_label = 'Test error'

            elif error_type == 'train_errors':
                errors = train_errors
                y_label = 'Train error'

            elif error_type == 'misclassification_test_errors':
                errors = misclassification_test_errors
                y_label = 'classification error'

            elif error_type == 'norms':
                errors = np.sqrt(f_norms)
                y_label = r'$\|\Theta - \Theta_0\|_F$'

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
        yerr=filtered_stds,
        color=color,
        fmt='o',  # square markers
        markersize=3,
        capsize=2.5,
        capthick=1,
        elinewidth=1.5,
        alpha=0.7
    )


#################################################








from multinomial_logistic.log_data.log_fp_tests import get_fp_misclassification_statistics


def plot_errors_vs_alpha(k, k_0, alpha_min_fp=1, alpha_max = 15,
                 alpha_min_emp=1, empirical_mean_std=True, 
                 empirical_mean_only=False, empirical_window=0.4):


    fp_results_two_classes_close = get_fp_statistics(k, k_0, two_classes_close=True)
    fp_results_symmetric = get_fp_statistics(k, k_0, non_symmetric=False, two_classes_close=False)
    fp_results_non_symmetric = get_fp_statistics(k, k_0, two_classes_close=False, non_symmetric=True)


    if fp_results_symmetric is None or fp_results_two_classes_close is None or fp_results_non_symmetric is None:
        print("No FP results found")
        return
    
    # Process input R_00 results
    alphas_fp_symmetric, test_errors_fp_symmetric, train_errors_fp_symmetric, F_norms_fp_symmetric, misclassification_test_errors_fp_symmetric = fp_results_symmetric
    #process two_classes_close results
    alphas_fp_two_classes_close, test_errors_fp_two_classes_close, train_errors_fp_two_classes_close, F_norms_fp_two_classes_close, misclassification_test_errors_fp_two_classes_close = fp_results_two_classes_close
    #process non_symmetric results
    alphas_fp_non_symmetric, test_errors_fp_non_symmetric, train_errors_fp_non_symmetric, F_norms_fp_non_symmetric, misclassification_test_errors_fp_non_symmetric = fp_results_non_symmetric


    # Filter FP results by alpha_min
 
    fp_mask_symmetric = (alphas_fp_symmetric >= alpha_min_fp) & (alphas_fp_symmetric < alpha_max)
    alphas_fp_symmetric = alphas_fp_symmetric[fp_mask_symmetric]
    test_errors_fp_symmetric = test_errors_fp_symmetric[fp_mask_symmetric]
    train_errors_fp_symmetric = train_errors_fp_symmetric[fp_mask_symmetric]
    F_norms_fp_symmetric = np.array(F_norms_fp_symmetric)[fp_mask_symmetric]
    misclassification_test_errors_fp_symmetric = misclassification_test_errors_fp_symmetric[fp_mask_symmetric]


    #filter two_classes_close data by alpha_min
    fp_mask_two_classes_close = (alphas_fp_two_classes_close >= alpha_min_fp) & (alphas_fp_two_classes_close < alpha_max)
    alphas_fp_two_classes_close = alphas_fp_two_classes_close[fp_mask_two_classes_close]
    test_errors_fp_two_classes_close = test_errors_fp_two_classes_close[fp_mask_two_classes_close]
    train_errors_fp_two_classes_close = train_errors_fp_two_classes_close[fp_mask_two_classes_close]
    F_norms_fp_two_classes_close = np.array(F_norms_fp_two_classes_close)[fp_mask_two_classes_close]
    misclassification_test_errors_fp_two_classes_close = misclassification_test_errors_fp_two_classes_close[fp_mask_two_classes_close]
    
    #filter non_symmetric data by alpha_min
    fp_mask_non_symmetric = (alphas_fp_non_symmetric >= alpha_min_fp) & (alphas_fp_non_symmetric < alpha_max)
    alphas_fp_non_symmetric = alphas_fp_non_symmetric[fp_mask_non_symmetric]
    test_errors_fp_non_symmetric = test_errors_fp_non_symmetric[fp_mask_non_symmetric]
    train_errors_fp_non_symmetric = train_errors_fp_non_symmetric[fp_mask_non_symmetric]
    F_norms_fp_non_symmetric = np.array(F_norms_fp_non_symmetric)[fp_mask_non_symmetric]
    misclassification_test_errors_fp_non_symmetric = misclassification_test_errors_fp_non_symmetric[fp_mask_non_symmetric]


    # Read MLE data for both d=50 and d=250
    mle_data_sets_symmetric = []
    mle_data_sets_two_classes_close = []
    mle_data_sets_non_symmetric = []
    d = 250
    try:
        with open(f'multinomial_logistic/log_data/newdata/mle_empirical/mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150.json', 'r') as f:
            mle_data_symmetric = json.load(f)

        with open(f'multinomial_logistic/log_data/newdata/mle_empirical/mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150_two_classes_close.json', 'r') as f:
            mle_data_two_classes_close = json.load(f)

        with open(f'multinomial_logistic/log_data/newdata/mle_empirical/mle_data_k{k}_k0{k_0}_lambda0_d{d}_ntrials150_non_symmetric.json', 'r') as f:
            mle_data_non_symmetric = json.load(f)
            
        # Extract empirical values for the symmetric case
        R_00 = np.array([[1,1/2], [1/2,1]])
        R_00_str = str(R_00.tolist())
        alpha_emp_vals = []
        test_error_emp = []
        train_error_emp = []
        misclassification_test_error_emp = []
        
        for alpha_str in mle_data_symmetric['results'][R_00_str].keys():
            alpha = float(alpha_str)
            if alpha >= alpha_min_emp and alpha < alpha_max:
                data = mle_data_symmetric['results'][R_00_str][alpha_str]
                if not data.get('div', False):  # Skip if diverged
                    alpha_emp_vals.append(alpha)
                    test_error_emp.append(data['test_errors'])
                    train_error_emp.append(data['train_errors'])
                    misclassification_test_error_emp.append(data['misclassification_test_errors'])
        mle_data_sets_symmetric.append({
            'd': 250,
            'alphas': alpha_emp_vals,
            'test_errors': test_error_emp,
            'train_errors': train_error_emp,
            'misclassification_test_errors': misclassification_test_error_emp
        })



        # two classes close emp
        R_00 = np.array([[1,0.9], [0.9,1]])
        R_00_str = str(R_00.tolist())
        alpha_emp_vals_two_classes_close = []
        test_error_emp_two_classes_close = []
        train_error_emp_two_classes_close = []
        misclassification_test_error_emp_two_classes_close = []

        
        
        for alpha_str in mle_data_two_classes_close['results'][R_00_str].keys():
            alpha = float(alpha_str)
            if alpha >= alpha_min_emp and alpha < alpha_max:
                data = mle_data_two_classes_close['results'][R_00_str][alpha_str]
                if not data.get('div', False):  # Skip if diverged
                    alpha_emp_vals_two_classes_close.append(alpha)
                    test_error_emp_two_classes_close.append(data['test_errors'])
                    train_error_emp_two_classes_close.append(data['train_errors'])
                    misclassification_test_error_emp_two_classes_close.append(data['misclassification_test_errors'])
        mle_data_sets_two_classes_close.append({
            'd': d,
            'alphas': alpha_emp_vals_two_classes_close,
            'test_errors': test_error_emp_two_classes_close,
            'train_errors': train_error_emp_two_classes_close,
            'misclassification_test_errors': misclassification_test_error_emp_two_classes_close
        })


        # extract empirical results for non_symmetric
        R_00 = np.array([[1,-1/2], [-1/2,1]])
        R_00_str = str(R_00.tolist())
        alpha_emp_vals_non_symmetric = []
        test_error_emp_non_symmetric = []
        train_error_emp_non_symmetric = []
        misclassification_test_error_emp_non_symmetric = []

        for alpha_str in mle_data_non_symmetric['results'][R_00_str].keys():
            alpha = float(alpha_str)
            if alpha >= alpha_min_emp and alpha < alpha_max:
                data = mle_data_non_symmetric['results'][R_00_str][alpha_str]
                if not data.get('div', False):  # Skip if diverged
                    alpha_emp_vals_non_symmetric.append(alpha)
                    test_error_emp_non_symmetric.append(data['test_errors'])
                    train_error_emp_non_symmetric.append(data['train_errors'])
                    misclassification_test_error_emp_non_symmetric.append(data['misclassification_test_errors'])
        mle_data_sets_non_symmetric.append({
            'd': d,
            'alphas': alpha_emp_vals_non_symmetric,
            'test_errors': test_error_emp_non_symmetric,
            'train_errors': train_error_emp_non_symmetric,
            'misclassification_test_errors': misclassification_test_error_emp_non_symmetric
        })

    except Exception as e:
        print(f"Error reading MLE data for d={d}: {e}")
    
    # Create figure directory if it doesn't exist
    fig_dir = os.path.join(os.path.dirname(__file__), "figures", "errors", "no_reg_alpha")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot settings
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

    
    # Create plots
    metrics = [
        ('test_error', 
         [test_errors_fp_symmetric, test_errors_fp_two_classes_close, test_errors_fp_non_symmetric],
         [alphas_fp_symmetric, alphas_fp_two_classes_close, alphas_fp_non_symmetric],
         [r'$\bold{R}_{00}= \bold{R}_{00}^{(1)}$', r'$\bold{R}_{00}= \bold{R}_{00}^{(2)}$', r'$\bold{R}_{00}= \bold{R}_{00}^{(3)}$']),
        ('train_error',
         [train_errors_fp_symmetric, train_errors_fp_two_classes_close, train_errors_fp_non_symmetric], 
         [alphas_fp_symmetric, alphas_fp_two_classes_close, alphas_fp_non_symmetric],
         [r'$\bold{R}_{00}= \bold{R}_{00}^{(1)}$', r'$\bold{R}_{00}= \bold{R}_{00}^{(2)}$', r'$\bold{R}_{00}= \bold{R}_{00}^{(3)}$']),
        ('misclassification_test_error',
         [misclassification_test_errors_fp_symmetric, misclassification_test_errors_fp_two_classes_close, misclassification_test_errors_fp_non_symmetric],
         [alphas_fp_symmetric, alphas_fp_two_classes_close, alphas_fp_non_symmetric],
        [r'$\bold{R}_{00}= \bold{R}_{00}^{(1)}$', r'$\bold{R}_{00}= \bold{R}_{00}^{(2)}$', r'$\bold{R}_{00}= \bold{R}_{00}^{(3)}$'])
    ]

    
    for metric_name, fp_values, alphas_fp, legends in metrics:
        print('plotting', metric_name)
        plt.figure(figsize=(10, 6))
        
        # Change this line to use proper LaTeX syntax for alpha
        alpha_min_test = alpha_min_fp
        alpha_max_test = alpha_max
        if metric_name == 'test_error':
            plt.title(r'Test Error vs $\alpha$')  # Use r prefix and proper LaTeX math mode
            alpha_min_test = 3
            alpha_max_test = 6
        elif metric_name == 'train_error':
            plt.title(r'Train Error vs $\alpha$')
        elif metric_name == 'misclassification_test_error':
            plt.title(r'Classification Error vs $\alpha$')
        elif metric_name == 'norms':
            plt.title(r'Norm vs $\alpha$')

        # Plot each FP solution separately
        for i in range(len(alphas_fp)):
            plt.plot(alphas_fp[i][(alphas_fp[i] >= alpha_min_test) & (alphas_fp[i] <= alpha_max_test)], 
                    fp_values[i][(alphas_fp[i] >= alpha_min_test) & (alphas_fp[i] <= alpha_max_test)], 
                    color=['darkblue', 'black', '#3E7893'][i], 
                    label=legends[i], 
                    linestyle='-', 
                    linewidth=2)
        
        # Plot empirical results for both d=50 and d=250
        for mle_data in mle_data_sets_symmetric:
            d = mle_data['d']
            # Convert lists to numpy arrays
            alphas = np.array(mle_data['alphas'])
            
            if metric_name == 'test_error':
                errors_list = [np.array([e for e in errors if e < 10]) for errors in mle_data['test_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                # Calculate standard error instead of standard deviation
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            elif metric_name == 'train_error':
                errors_list = [np.array([e for e in errors if e < 5]) for errors in mle_data['train_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            elif metric_name == 'misclassification_test_error':
                errors_list = [np.array([e for e in errors if e < 10]) for errors in mle_data['misclassification_test_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            
            # Convert means and stds to numpy arrays
            emp_means = np.array(emp_means)
            emp_stds = np.array(emp_stds)
            
            # Now use boolean indexing with numpy arrays
            mask = (alphas >= alpha_min_emp) & (alphas <= alpha_max_test)
            plt.errorbar(alphas[mask], 
                        emp_means[mask], 
                        yerr=emp_stds[mask],
                        color='darkblue', fmt='s',
                        markersize=3, alpha=0.7,
                        capsize=3, elinewidth=1)
            

        # plot empirical result for nonsymmetric    
        for mle_data in mle_data_sets_non_symmetric:
            d = mle_data['d']
            # Convert lists to numpy arrays
            alphas = np.array(mle_data['alphas'])
            
            if metric_name == 'test_error':
                errors_list = [np.array([e for e in errors if e < 10]) for errors in mle_data['test_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            elif metric_name == 'train_error':
                errors_list = [np.array([e for e in errors if e < 5]) for errors in mle_data['train_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            elif metric_name == 'misclassification_test_error':
                errors_list = [np.array([e for e in errors if e < 10]) for errors in mle_data['misclassification_test_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            
            emp_means = np.array(emp_means)
            emp_stds = np.array(emp_stds)
            
            # Now use boolean indexing with numpy arrays
            mask = (alphas >= alpha_min_emp) & (alphas <= alpha_max_test)
            plt.errorbar(alphas[mask], 
                        emp_means[mask], 
                        yerr=emp_stds[mask],
                        color='#3E7893', fmt='s',
                        markersize=3, alpha=0.7,
                        capsize=3, elinewidth=1)
            
        # Plot empirical results for both d=50 and d=250 for two classes close
        for mle_data in mle_data_sets_two_classes_close:
            d = mle_data['d']
            # Convert lists to numpy arrays
            alphas = np.array(mle_data['alphas'])
            
            if metric_name == 'test_error':
                y_label = r'Test error'
                errors_list = [np.array([e for e in errors if e < 10]) for errors in mle_data['test_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            elif metric_name == 'train_error':
                y_label = r'Train error'
                errors_list = [np.array([e for e in errors if e < 5]) for errors in mle_data['train_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            elif metric_name == 'misclassification_test_error':
                y_label = r'Classification error'
                errors_list = [np.array([e for e in errors if e < 10]) for errors in mle_data['misclassification_test_errors']]
                emp_means = [np.mean(errors) for errors in errors_list]
                emp_stds = [np.std(errors) / np.sqrt(len(errors)) for errors in errors_list]
            
            emp_means = np.array(emp_means)
            emp_stds = np.array(emp_stds)
            
            # Now use boolean indexing with numpy arrays
            mask = (alphas >= alpha_min_emp) & (alphas <= alpha_max_test)
            plt.errorbar(alphas[mask], 
                        emp_means[mask], 
                        yerr=emp_stds[mask],
                        color='black', fmt='s',
                        markersize=3, alpha=0.7,
                        capsize=3, elinewidth=1)
        



        plt.xlabel(r'$\alpha$', fontsize=22)
        plt.ylabel(y_label, fontsize=22)  # Increased font size for y-label
        plt.title('')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        filename = (f'{metric_name}_k{k}_k0{k_0}.pdf')
        plt.savefig(os.path.join(fig_dir, filename))
        plt.close()
        print(f'saved the errors plot at {os.path.join(fig_dir, filename)}')




















def plot_density(k,
                 k_0, 
                 R_00, 
                 alpha_target, 
                 eigenvalues=None, 
                 density_lower_bound=0, 
                 d=250, 
                 lambda_reg=0,
                 save_path=None, 
                 emp_histogram=True, 
                 emp_bins=80, 
                 z_real_lower_bound=0.014, 
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

    # Add a zero point at just below the lower bound to ensure the curve starts at zero
    if clean_data_for_5:
        densities_filtered = densities_filtered[10:]
        z_reals_filtered = z_reals_filtered[10:]
        densities_filtered[0] = 0
    if clean_data_for_10:
        print('cleaning data for 10')
        densities_filtered = densities_filtered[2:]
        z_reals_filtered = z_reals_filtered[2:]
        densities_filtered[-1] = 0
        densities_filtered[0] = 0

        z_reals_filtered = np.delete(z_reals_filtered, [7,16,20,22])
        densities_filtered = np.delete(densities_filtered, [7,16,20,22])
        z_reals_filtered = np.delete(z_reals_filtered, [14,21,23])
        densities_filtered = np.delete(densities_filtered, [14,21,23])
        z_reals_filtered = np.delete(z_reals_filtered, [19,49])
        densities_filtered = np.delete(densities_filtered, [19,49])

    if clean_data_for_3:
        densities_filtered = densities_filtered[:-2]
        z_reals_filtered = z_reals_filtered[:-2]
        #z_reals_filtered = np.insert(z_reals_filtered, 0, 0.001)
        densities_filtered[0] = 0
        densities_filtered[-1] = 0
    if clean_data_for_20:
        
        densities_filtered = densities_filtered[2:-3]
        z_reals_filtered = z_reals_filtered[2:-3]
        densities_filtered[0] = 0
        densities_filtered[-1] = 0

        z_reals_filtered = np.delete(z_reals_filtered, [9,28])
        densities_filtered = np.delete(densities_filtered, [9,28])



    for i in range(len(z_reals_filtered)):
        print(f'[index {i}]: z:', z_reals_filtered[i], 'density:', densities_filtered[i])


    # -----------------------------
    # 3. Create and style the figure
    # -----------------------------
    fig, ax = plt.subplots()

    # Plot the theoretical density curve
    ax.plot(
        z_reals_filtered, 
        densities_filtered, 
        color='darkblue', 
        linewidth=2.4, 
        label=r'$\mu_{\star}(\nu_{\mathrm{opt}})$'
    )

    # Style the axes
    #ax.spines['top'].set_visible(True)
    #ax.spines['right'].set_visible(True)

    # -----------------------------------------
    # 4. Add empirical histogram if requested
    # -----------------------------------------
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
        fig_dir = os.path.join(os.path.dirname(__file__), "figures", "fixed_density")
        os.makedirs(fig_dir, exist_ok=True)
        filename = (f"density_k{k}_k0{k_0}_R00{R_00[0,1]}"
                    f"_alpha{alpha_target:.2f}_lambda{lambda_reg}"
                    f"_emp{int(emp_histogram)}_d{d}.pdf")
        save_path = os.path.join(fig_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Info] Plot saved to {save_path}")

    # Close the figure to free up memory
    plt.close(fig)