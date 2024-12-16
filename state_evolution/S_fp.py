import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

from state_evolution.functions import score_batched, score_jacobian_batched

from multinomial_logistic.fixed_point_system.fp_integrands import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_inv, batched_mlogit, batched_scalar_mult    
from multinomial_logistic.utils import batched_mlogit_jacobian, batched_normal_basis, batched_product
from multinomial_logistic.prox import prox_fp_iteration
from cubature import cubature

"""
Here we calculate the state evolution for the fixed point system
E[ (I_k + S^* J P(prox(g + yS; S))^{-1} ) ] = (1-1/alpha)I_k
"""



def S_fp_solver_new(R_00, schur_t, A_t, alpha, k, k_0):
    print("in S_fp_solver... ")
    S_t = np.eye(k)
    canonical = False
    for t in range(1000):
        integrand = integration(_S_fp_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0, canonical).reshape(k,k)
        S_next = 1/alpha * np.linalg.inv(np.eye(k) - integrand) @ S_t
        print(f'     **S_fp_solver iteration {t+1} finished with S={S_next.flatten()}**')
        if np.linalg.norm(S_t - S_next) < 1e-3:
            print(f'     **S_fp_solver converged in {t+1} iterations**')
            break
        S_t = S_next
    return S_t

"""
def S_fp_solver(R_00, schur_t, A_t, alpha, k, k_0):
    #print("in S_fp_solver... ")
    S_init_canonical = np.eye(k).flatten()

    solution, infodict, ier, mesg = fsolve(S_fp_equation, S_init_canonical, 
                                         args=(R_00, schur_t, A_t, alpha, k, k_0,),
                                         xtol=1e-10, epsfcn=1e-10, 
                                         full_output=True)
    
    print("            Computing S...")
    print("            Termination message:", mesg)
    print("            Number of function evaluations:", infodict['nfev'])
    print("            Final residuals:", infodict['fvec'])

    S_canonical = solution.reshape((k,k))
    S = S_canonical @ S_canonical.T
    return S
"""

def S_fp_equation(S_flat_canonical, R_00, schur_t, A_t, alpha, k, k_0):
    print("   in S_fp_equation...")
    S_canonical  = S_flat_canonical.reshape((k, k))
    eq_flat = integration(_S_fp_integrand, S_canonical=S_canonical, R_00=R_00, schur_t=schur_t, A_t=A_t, alpha=alpha, k=k, k_0=k_0)
    #print('  fp_eq is computed:', np.array(eq_flat.reshape(k,k) - ((1-1/alpha) * np.eye(k))).flatten())
    print('residual:', np.array(eq_flat.reshape(k,k) - ((1 - 1/alpha) * np.eye(k))).flatten())
    return np.array(eq_flat.reshape(k,k) - ((1 - 1/alpha) * np.eye(k))).flatten()

"""
*************************************************************************************
nasty integrands
"""

def _S_fp_integrand(Z_batch, S_canonical, R_00, schur_t, A_t, alpha, k, k_0, canonical):
    N = Z_batch.shape[0]
    if canonical:
        S = S_canonical @ S_canonical.T # to ensure S is positive semidefinite
    else:
        S = S_canonical
    schur_root = sqrtm(schur_t)
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root, alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S=S) # prox(g + yS; S)
        score_jacobian_batch = score_jacobian_batched(prox_g_batch, S, k) # (I + S @ Jp(prox(g + yS; S)))^{-1}
        integrand += batched_scalar_mult(score_jacobian_batch, prob_y_batch[:, i]) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened



def integration(integrand, S_canonical, R_00, schur_t, A_t, alpha, k, k_0, canonical):
    #print('integrating... ')
    fdim = k*k
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(S_canonical, R_00, schur_t, A_t, alpha, k, k_0, canonical), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-8]*ndim, xmax=[8]*ndim, abserr = 1e-6, relerr=1e-6,
                                  maxEval=100000, norm=2)
    if err.any() > 1e-2:
        print('     **[Warning] integration in the S_fp error is too large**')
    #print('done integrating')
    return expectations






