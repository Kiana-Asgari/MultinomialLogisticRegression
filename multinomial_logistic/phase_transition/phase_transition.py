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

import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigsh
from scipy.stats import multivariate_normal
from cubature import cubature
from scipy.optimize import minimize
import os



from multinomial_logistic.utils import batched_mlogit, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import batched_mult
from multinomial_logistic.prox import prox_fp_iteration
import matplotlib.pyplot as plt
import sys as sys


def plot_phase_transition(k, k_0, seed=58):
    print(f"Plotting phase transition for k={k}")
    R_values = np.linspace(0.01, 20, 150)
    kappa_values = np.empty_like(R_values)
    
    for i, R in enumerate(R_values):
        R_00 = np.eye(k_0) * R
        C_opt, alpha_opt = phase_transition_minimization(R_00, k, k_0, seed)
        kappa_values[i] = 1/alpha_opt

    plt.figure(figsize=(10, 6))
    plt.plot(kappa_values, R_values, label=f'k={k}, k_0={k_0}')
    plt.xlabel('kappa')
    plt.ylabel('R')
    plt.title('Phase Transition')
    plt.legend()
    
    # Save the plot
    save_path = f'multinomial_logistic/phase_transition/plots/phase_transition_k{k}_k0{k_0}.png'
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved to: {save_path}")


def phase_transition_minimization(R_00, k, k_0, seed=58):
    # min_{C in R^{k*k}} E[f(C, R_00)]
    print("  ...Starting to compute the phase transition for k = ", k, " and R_00 = ", R_00.flatten())
    np.random.seed(seed)
    C0 = np.eye(k).flatten()

    # Use SciPy's minimize
    res = minimize(phase_transition_equation, C0, args=(R_00, k, k_0,), method='BFGS')

    # Extract solution
    C_opt_flattened = res.x
    C_opt = C_opt_flattened.reshape((k, k))
    alpha_opt = 1/res.fun

    print("Done computing the phase transition for k = ", k, " and R_00 = ", R_00.flatten())
    print("  ...SciPy Optimization Results:")
    print("  ...Optimal value of the objective:", res.fun)
    print("  ...Optimal alpha:", alpha_opt)
    print("  ...Optimal matrix C:\n", C_opt)
    print("  ...Optimization Error Metrics:")
    print("  ...Gradient norm:", np.linalg.norm(res.jac))  # Gradient norm at solution
    print("  ...Number of iterations:", res.nit)           # Number of iterations
    print("  ...Number of function evaluations:", res.nfev) # Number of function evaluations
    print("  ...Converged:", res.success)
    print("  ...Message:", res.message)

    return C_opt, alpha_opt





def phase_transition_equation(C_flattened, R_00, k, k_0):
    # E[f(C, R_00)] 
    C = C_flattened.reshape((k, k))
    integrand = integration(_integrand, C, R_00, k, k_0)
    max_eigenvalue = eigsh(integrand, k=1, which='LM', return_eigenvectors=False)[0]
    return max_eigenvalue

def _integrand(G_batch, C, R_00, k, k_0):
    # f(C, R_00) = (Z - C@G_0 - G) @ (Z - C@G_0 - G).T

    N = G_batch.shape[0]
    g_batch, g_0_batch = split_gaussian_vectors(G_batch, k, k_0, R_00) # g_0~ N(0, R_00), g~N(0, I_k)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(G_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N)
        Z_batch = _optimal_random_variable(g_batch, g_0_batch, y_batch, C)
        FP_batch = Z_batch - np.einsum('ij,Nj->Ni', C, g_0_batch) - g_batch
        FP_batch = np.einsum('Ni,Nj->Nij', FP_batch, FP_batch)
        integrand += batched_scalar_mult(FP_batch, prob_y_batch[:, i])   
 

    integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened


def _optimal_random_variable(g_batch, g_0_batch, y_batch, C):
    # Z = sum_{j=1}^k [(C@G_0 + G)_j]_{+} @e_j 1_{y = e_j}
    #     - sum_{j=1}^k [(C@G_0 + G)_j]_{-} @e_j 1_{y = 0};

    N = y_batch.shape[0]
    composition = np.einsum('ij,Nj->Ni', C, g_0_batch) + g_batch # N*k
    y_base_class = np.ones(N) - np.sum(y_batch, axis=1) # =1 if y is the base class, N

    Z_batch_positive = np.maximum(np.einsum('Ni,Ni->N', composition, y_batch), 0) # N
    Z_batch_negative = np.minimum(composition, 0) # N*k

    Z_batch = np.einsum('N,Ni->Ni', Z_batch_positive, y_batch) \
              - np.einsum('Ni,N->Ni', Z_batch_negative, y_base_class)# N*k

    return Z_batch # N*k




def split_gaussian_vectors(G_batch, k, k_0, R_00):
    G_top  = G_batch[:,:k] # (N, k)
    G_bottom = G_batch[:,-k_0:] # (N, k_0)
    g = G_top                                 # g ~ N(0, I_k)
    g_0 = batched_mult(sqrtm(R_00), G_bottom) # g_0 ~ N(0, R_00)
    return g, g_0


def integration(integrand, C, R_00, k, k_0):
    fdim = k*k
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(C, R_00, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-3.4]*ndim, xmax=[3.4]*ndim, abserr=1e-5,
                                  maxEval= 250_000, norm=2)
    #print('     integration error: ', err)
    #for e in err:
    #   if e > 1e-3:
    #     print('     **[Warning] state evolution integration error is too large**')
    #     break
    #print('     done integrating')
    return expectations.reshape(k, k)



