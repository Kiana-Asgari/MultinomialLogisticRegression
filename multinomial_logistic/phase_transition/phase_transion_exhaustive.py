import numpy as np
import matplotlib.pyplot as plt
import os

from state_evolution.full_recursion import state_evolution_full_recursion


def plot_phase_transition_exhaustive(R_00, k, k_0, r_min = 0, r_max = 10, \
                                    tol=1e-2, max_iter=100):

    R_values = np.linspace(r_min, r_max, 100)
    kappa_values = []
    divergence_values = []
    alpha = 1.1

    for r in R_values:
        R = r * R_00
        alpha = phase_transition_exhaustive_search(R_00=R, k=k, k_0=k_0, \
                                                    alpha_min=alpha, alpha_max=max(alpha+2, 4), \
                                                    tol=tol, max_iter=max_iter)
        kappa_values.append(1/alpha)

    plt.figure(figsize=(6, 6))
    
    # Set axes to start from minimum kappa value
    plt.xlim(min(kappa_values), max(kappa_values) * 1.1)  # Add 10% padding on the right
    plt.ylim(0, max(R_values) * 1.1)      # Add 10% padding on top
    
    # Fill areas with different shades of blue
    plt.fill_between(kappa_values, 0, R_values, color='lightblue', alpha=0.5, label='Below curve')
    plt.fill_between(kappa_values, R_values, plt.ylim()[1], color='darkblue', alpha=0.3, label='Above curve')
    
    # Plot the transition line
    plt.plot(kappa_values, R_values, 'k-', linewidth=1, label=f'k={k}, k_0={k_0}')
    
    # Remove frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add text labels in each region
    # Calculate positions for text (roughly center of each region)
    x_pos_below = np.mean(kappa_values) * 0.6  # Move left
    x_pos_above = np.mean(kappa_values) * 1.4  # Move right
    y_pos_below = np.mean(R_values) * 0.3      # Move down
    y_pos_above = np.mean(R_values) * 1.7      # Move up

    plt.text(x_pos_below, y_pos_below, 'MLE exists', 
             color='black', 
             fontsize=12, 
             ha='center', 
             va='center')
    
    plt.text(x_pos_above, y_pos_above, 'MLE does not exist', 
             color='black', 
             fontsize=12, 
             ha='center', 
             va='center')
    
    plt.xlabel('1/alpha')
    plt.ylabel('r_0')
    plt.title(f'Phase Transition for k={k}, R_00 = r_0^2*I_k')
    #plt.legend()
    
    # Save the plot
    save_path = f'multinomial_logistic/phase_transition/plots/k{k}try1.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Plot saved to: {save_path}")
        
def phase_transition_exhaustive_search(R_00, k, k_0,\
                                        alpha_min=2, alpha_max=2.7, tol=1e-2, max_iter=100):
    alpha = alpha_max
    schur = R_00
    R_01 = np.zeros((k_0,k))
    S = np.eye(k)

    for i in range(max_iter):
        schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=schur, R_01_0=R_01, S_0=S, \
                                        lambda_reg=0, alpha=alpha, k=k, k_0=k_0, tol=tol)
        if divergence:
            break

        norm = max(np.linalg.norm(R_01), np.linalg.norm(S), np.linalg.norm(schur))
        print(f"Alpha: {alpha}, Divergence: {divergence}, norm: {norm}")


        alpha -= min(max(10/norm, 1e-1), 0.4)

    return alpha
