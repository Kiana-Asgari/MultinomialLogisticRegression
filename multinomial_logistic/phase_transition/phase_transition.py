"""
computing the phase transition for the multinomial logistic model
    *************************************************************
    *   1/alpha I_k   >  min_{C in R^{k*k}} E[f(C, R_00)],         *
    *************************************************************
where G_0 ~ N(0, R_00), G ~ N(0, I_k),

        Z = sum_{j=1}^k [(C@G_0 + G)_j]_{+} @e_j 1_{y = e_j}
            - sum_{j=1}^k [(C@G_0 + G)_j]_{-} @e_j 1_{y = 0};

        f(C, R_00) = (Z - C@G_0 - G) @ (Z - C@G_0 - G).T
"""
import torch

import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigsh
from scipy.stats import multivariate_normal
from cubature import cubature
from scipy.optimize import minimize
import os
from scipy.linalg import eigh



from multinomial_logistic.utils import batched_mlogit, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import batched_mult
from multinomial_logistic.prox import prox_fp_iteration
import matplotlib.pyplot as plt
import sys as sys


def plot_phase_transition(k, k_0, seed=58):
    print(f"Plotting phase transition for k={k}")
    R_values = np.linspace(0, 20, 30)
    kappa_values = np.empty_like(R_values)

    C0 = np.zeros(k*k)
    Lambda_0 = np.ones(k-1)
    

    
    for i, R in enumerate(R_values):
        if k == 2:
            R_00 = np.diag([R*R, R*R])
        else:
            R_00 = R * np.eye(k_0) * R
        C0, Lambda_0,alpha_opt = phase_transition_minimization(R_00, k, k_0, C0.flatten(), Lambda_0, seed)
        kappa_values[i] = 1/alpha_opt

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


def phase_transition_minimization(R_00, k, k_0, C0, Lambda, seed=58):
    # min_{C in R^{k*k}} E[f(C, R_00)]
    print("  ...Starting to compute the phase transition for k = ", k, " and R_00 = ", R_00.flatten())
    np.random.seed(seed)
    minimization_params = np.concatenate([C0.flatten(), Lambda.flatten()]) # k*k + k-1

    # Use SciPy's minimize
    res = minimize(phase_transition_equation, minimization_params, args=(R_00, k, k_0,), method='CG',\
                    options={'maxiter': 10, 'gtol': 1e-2, 'eps': 1e-4})

    # Extract solution
    Lambda_opt = res.x[-(k_0-1):]
    C_opt_flattened = res.x[:-(k_0-1)]
    C_opt = C_opt_flattened.reshape((k, k))
    alpha_opt = 1/res.fun

    print("Done computing the phase transition for k = ", k, " and R_00 = ", R_00.flatten())
    print("  ...SciPy Optimization Results:")
    print("  ...Number of iterations:", res.nit)
    print("  ...Optimal value of the objective:", res.fun)
    print("  ...Optimal alpha:", alpha_opt)
    print("  ...Optimal matrix C:\n", C_opt)
    print("  ...Optimal Lambda:\n", Lambda_opt)


    return C_opt, Lambda_opt, alpha_opt





def phase_transition_equation(params_flattened, R_00, k, k_0):
    # E[f(C, R_00)] 
    #print("  ...Starting to compute the phase transition equation C=", C_flattened)
    C = params_flattened[:-(k_0-1)].reshape((k, k))
    Lambda_flattened = params_flattened[-(k_0-1):]
    Lambda = np.diag(np.concatenate([[1], Lambda_flattened]))
    Lambda_inv = np.diag(np.concatenate([[1], 1/Lambda_flattened]))
    
    integrand = Lambda_inv @ integration(_integrand, C, Lambda, R_00, k, k_0) @ Lambda_inv
    max_eigenvalue = np.linalg.norm(integrand, ord=2)
    #print("         ...lambda max guess = ", max_eigenvalue )
    F_norm = 1/k*np.linalg.norm(integrand)
    print("         ...max_eigenvalue:", max_eigenvalue, 'F_norm:', F_norm)
    print('                 integrand:', integrand.flatten())
    return max_eigenvalue

def _integrand(G_batch, C, Lambda, R_00, k, k_0):
    # f(C, R_00) = (Z - C@G_0 - G) @ (Z - C@G_0 - G).T

    N = G_batch.shape[0]
    g_batch, g_0_batch = split_gaussian_vectors(G_batch, k, k_0, R_00) # g_0~ N(0, R_00), g~N(0, I_k)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(G_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N)
        composition = np.einsum('ij,Nj->Ni', C, g_0_batch) + np.einsum('ij,Nj->Ni', Lambda, g_batch) # N*k
        Z_batch = _optimal_random_variable(composition, y_batch, C)
        FP_batch = Z_batch - composition # z - Cg_0 - g
        FP_batch = np.einsum('Ni,Nj->Nij', FP_batch, FP_batch) # (Z - Cg_0 - g)(Z - Cg_0 - g).T
        #FP_batch = test_integrand(g_batch, g_0_batch, y_batch, C)
        integrand += batched_scalar_mult(FP_batch, prob_y_batch[:, i])   
 

    integrand = batched_scalar_mult(integrand, pdf) # (Z - Cg_0 - g)(Z - Cg_0 - g).T * p(g,g_0)

    if k == 1:
        return integrand.reshape(-1) #flattened  
    return integrand.reshape(N, -1)  #flattened




def _optimal_random_variable(composition, y_batch, C):
    # Z = sum_{j=1}^k [(C@G_0 + G)_j]_{+} @e_j 1_{y = e_j}
    #     - sum_{j=1}^k [(C@G_0 + G)_j]_{-} @e_j 1_{y = 0};

    N = y_batch.shape[0]
    y_base_class = np.ones(N) - np.sum(y_batch, axis=1) # =1 if y is the base class, N
    y_batch_flipped = 1-y_batch
    
    composition_positive = np.maximum(composition, 0) # N*k
    composition_negative = np.minimum(composition, 0) # N*k
    z_batch = np.einsum('Ni,Ni->Ni', composition_positive, y_batch) \
              + np.einsum('Ni,Ni->Ni', composition_negative, y_batch_flipped) # N*k
    return z_batch # N*k

    #Z_batch_positive = np.maximum(np.einsum('Ni,Ni->N', composition, y_batch), 0) # N
    #Z_batch_negative = np.minimum(composition, 0) # N*k

    #Z_batch = np.einsum('N,Ni->Ni', Z_batch_positive, y_batch) \
    #          + np.einsum('Ni,N->Ni', Z_batch_negative, y_base_class)# N*k

    #return Z_batch # N*k




def split_gaussian_vectors(G_batch, k, k_0, R_00):
    G_top  = G_batch[:,:k] # (N, k)
    G_bottom = G_batch[:,-k_0:] # (N, k_0)
    g = G_top                                 # g ~ N(0, I_k)
    g_0 = batched_mult(sqrtm(R_00), G_bottom) # g_0 ~ N(0, R_00)
    return g, g_0


def integration(integrand, C, Lambda, R_00, k, k_0):
    fdim = k*k
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(C, Lambda, R_00, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-3.4]*ndim, xmax=[3.4]*ndim, abserr=1e-6,
                                  maxEval= 250_000, norm=2)
    #print('     integration error: ', err)
    for e in err:
       if e > 1e-3:
         print('     **[Warning] state evolution integration error is too large**', e)
         break
    #print('     done integrating')
    return expectations.reshape(k, k)



""""
def test_integrand(g_batch, g_0_batch, y_batch, C):
    N = y_batch.shape[0]
    composition = np.einsum('ij,Nj->Ni', C, g_0_batch) # N*k
    y_base_class = np.ones(N) - np.sum(y_batch, axis=1) # =1 if y is the base class, N
    
    negative_composition = np.maximum(composition + g_batch, 0) # N*k
    negative_composition_y = np.einsum('Ni,N->Ni', negative_composition, y_base_class) # N*k
    negative_composition_squared = np.einsum('Ni,Nj->Nij', negative_composition, negative_composition) # N*k*k
    
    positive_composition = np.minimum(composition + g_batch, 0) # N*k
    positive_composition_squared = np.einsum('Ni,Nj->Nij', positive_composition, positive_composition) # N*k*k
    positive_composition_y = np.einsum('Ni,N->Ni', positive_composition, y_batch.flatten()) # N*k
    #phase_equation = np.einsum('Nij,N->Nij', positive_composition_squared, y_batch.flatten()) \
    #          + np.einsum('Nij,N->Nij', negative_composition_squared, y_base_class)# N*k


    z_batch = np.einsum('Ni,N->Ni', positive_composition_y, y_batch.flatten())\
                + np.einsum('Ni,N->Ni', negative_composition_y, y_base_class) # N*k
    #phase_equation = np.einsum('Ni, Nj->Nij',negative_composition_y+positive_composition_y,\
    #                            negative_composition_y+positive_composition_y)
    phase_equation = np.einsum('Ni, Nj->Nij',z_batch - composition - g_batch,\
                                z_batch - composition - g_batch)

    return phase_equation
"""
