import numpy as np

from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature


from state_evolution.functions import score_batched, score_jacobian_batched, MP_batched
from multinomial_logistic.utils import batched_mlogit, batched_outer, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import coloring_transform, batched_mult
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system
from multinomial_logistic.utils import wrapper, batched_mlogit_jacobian, batched_inv
from multinomial_logistic.evaluation.utils import  plot_distribution, custom_linspace

from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import esd_empirical




"""
Solving the equation: E_\nu [(I + D\bar S)^{-1} - zI]^{-1} = 
                                                        1/ alpha * bar S
For bar S \in {R}^{k * k},
 nu = Law(Prox(g + Sy; S), g_0), 
       where (g,g_0) ~ N(0, R)
       S, R are the solution of the fixed point equations

D(v,u) = batched_mlogit_jacobian(Prox(g + Sy; S)) 

"""

def recover_density( R_00, alpha, k, k_0, z_imag=1e-3,\
                    save_path='multinomial_logistic/data/density/2_classes'):
    """
    Recovers the full density indise the support of the ESD 
    from the solution of the fixed point equation
    """
    print(' Starting to recover density...')

    z_real_values = custom_linspace(0.002, 0.6, n_points=150)
    density_curve  = np.zeros_like(z_real_values)   
    legends = ['density']
    max_density = 0
    last_MP_S = np.complex128(np.eye(k))

    schur, R_01, S, _ = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k,k)),\
                                            lambda_reg=0, alpha=alpha, k=k, k_0=k)
    A = R_01 @ np.linalg.inv(sqrtm(R_00))

    for i, z_real in enumerate(z_real_values):
        new_MP_S, density = stieltjes_inversion(R_00, schur, A, S, z_real=z_real, z_imag=z_imag,\
                                       alpha=alpha, k=k, k_0=k_0, last_MP_S=last_MP_S)
        if density > max_density:
            max_density = density
        print('     z_real = ', z_real, '     density = ', density)
        
        density_curve[i] = density
        last_MP_S = new_MP_S
        if density < 0.5*1e-3 * max_density:
            print('     density is too small, stopping...')
            break
    _, empirical_density = esd_empirical(alpha=alpha, k=k, lambda_reg=0, R_00=R_00, max_iter=100, d=250)

    title = f"ESD theoretical, number of class={k+1:d},R_00= ({R_00}), alpha={alpha}"
    name = f"ESD_theoretical_nclass={k+1:d}_R_00={R_00}_alpha={alpha}" 
    print(' density_curve = ', density_curve.shape)
    print(' z_real_values = ', z_real_values.shape)
    plot_distribution(density=density_curve, empirical_density=empirical_density,\
                       z_values=z_real_values, title=title, \
                       x_label="eigenvalue", y_label="density", \
                       name=name, save_path=save_path)

    return density


def stieltjes_inversion(R_00, schur, A, S, z_real, alpha, k, k_0, z_imag, last_MP_S=None):
    """
    Solves the fixed point equation: E_\nu [(I + D\bar S)^{-1} - z_MP I]^{-1} = 1/ alpha * \bar S
    """
    MP_S, stieltjes_transform = MP_iteration(R_00=R_00, schur=schur, A=A, S=S, z_real=z_real, 
                                          last_MP_S=last_MP_S, z_imag=z_imag, alpha=alpha, k=k, k_0=k_0)
    density = stieltjes_transform.imag / np.pi
    print('     density at z_real = ', z_real, '     density = ', density)
    return MP_S, density
    



def MP_iteration(R_00, schur, A, S, z_real, z_imag, alpha, k, k_0, \
              last_MP_S=None, tol=1e-3, max_iter=350):
    """
    Solves the fixed point equation: E_\nu [(I + D\bar S)^{-1} - z_MP I]^{-1} = 1/ alpha * \bar S
    """
    if last_MP_S is None:
        MP_S_current_inv = np.complex128(np.eye(k))
    else:
        MP_S_current_inv = batched_inv(last_MP_S)
    err = 0

    print(f'*********[Starting MP iteration]... for z_real = {z_real+z_imag*1j}, starting MP_S_inv = {MP_S_current_inv.flatten()}')
    
    for t in range(max_iter):  
        #print(f'     iteration {t}...')

        #MP_S_next = alpha * _MP_F_equation(R_00=R_00, schur=schur, A=A, S=S,\
        #                                    z_real=z_real, z_imag=z_imag, MP_S=MP_S_current, alpha=alpha, k=k, k_0=k_0)
        MP_S_next_inv =  alpha * _MP_F_equation(R_00=R_00, schur=schur, A=A, S=S,\
                                            z_real=z_real, z_imag=z_imag, MP_S_inv=MP_S_current_inv, alpha=alpha, k=k, k_0=k_0)
        MP_S_next = batched_inv(MP_S_next_inv)
        MP_S_current = batched_inv(MP_S_current_inv)
        stieltjes_transform_current =  1/k *alpha* np.trace(MP_S_current)
        stieltjes_transform_next = 1/k *alpha* np.trace(MP_S_next)
        #stieltjes_transform_current =  1/k * np.trace(MP_S_current)
        #stieltjes_transform_next = 1/k * np.trace(MP_S_next)

        err_image = np.linalg.norm(MP_S_next.imag - MP_S_current.imag)/np.linalg.norm(MP_S_next.imag)
        err_real = np.linalg.norm(MP_S_next.real - MP_S_current.real)/np.linalg.norm(MP_S_next.real)
        err = np.max([err_image, err_real])
        print('    iteration=',t,' err=',err,'     MP_img = ', MP_S_next.imag.flatten())
        MP_S_current_inv = MP_S_next_inv
 
        if stieltjes_transform_next.imag < 0:
            print('     WARN: stieltjes_transform_next.imag < 0')
            break

        if err < tol:
            break


    stieltjes_transform = alpha *1/k * np.trace(MP_S_current)
    #stieltjes_transform = 1/k * np.trace(MP_S_current)
    print('     stieltjes_transform = ', stieltjes_transform)
    print('                     Err = ', err)
    print('                 MP iteration finished with iteration ', t, '*******************')
    return MP_S_current, stieltjes_transform



def _MP_F_equation( R_00, schur, A, S, z_real, MP_S_inv, alpha, k, k_0, z_imag):
    """
    Returns F(S;\nu) = {E_\nu [(I + D*MP_S)^{-1}*D - z_MP I]}^{-1} 
    """
    z = np.complex128(z_real + z_imag*1j)

    expectation = complex_integration(_MP_integrand, R_00=R_00, schur=schur, A=A, S=S,\
                               MP_S_inv=MP_S_inv, alpha=alpha, k=k, k_0=k_0)
    #return np.linalg.inv(expectation - z * np.eye(k))
    return expectation - z * np.eye(k)




def _MP_integrand(G_batch, R_00, schur, A, S, MP_S_inv, alpha, k, k_0):
    """
    Returns:[I_k + MP_S @ Jp(prox(g + yS; S))]^{-1} @ Jp(prox(g + yS; S)),
    MP_S is complex valued k*k matrix and is the solution of the MP 
    S is real valued k*k matrix given by the solution of the fixed point equation
    """
    N = G_batch.shape[0]
    schur_root = sqrtm(schur)

    # Coloring transform
    g_batch, g_0_batch = coloring_transform(G_batch, A=A, R_00=R_00, schur_root=schur_root  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(G_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k), dtype=np.complex128)

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch, prox_div = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S) # prox(g + yS; S)
        if prox_div:
            print('     WARN: prox divergence in _MP_integrand')
        MP_batch = MP_batched(V_batch=prox_g_batch, MP_S_inv=MP_S_inv, k=k) # N*k*k (I_k + S @ Jp(prox(g + yS; S)))^{-1} @ Jp(prox(g + yS; S))
        integrand += batched_scalar_mult(MP_batch, prob_y_batch[:, i])    

    integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)
   
    
    integrand_flat = integrand.reshape(N, -1) #flattened, N*k^2 

    # to deal with complex integration, we split the real and imaginary parts of the integrand
    return np.hstack([integrand_flat.real, integrand_flat.imag])  # N*2k^2


#####################################################################################

def complex_integration(integrand, R_00, schur, A, S, MP_S_inv, alpha, k, k_0):
    fdim = 2*k*k
    ndim = k+k_0

    expectations, err = cubature(integrand, args=(R_00, schur, A, S, MP_S_inv, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim=fdim,
                                  xmin=[-3.4]*ndim, xmax=[3.4]*ndim, 
                                  abserr=1e-5,
                                  maxEval=300_000, norm=1)
    
    # Check errors element by element and print problematic components
    problem_indices = np.where(err > 1e-4)[0]
    if len(problem_indices) > 0:
        print(' WARN: Integration errors exceeded threshold for components:')
        for idx in problem_indices:
            if idx < k*k:
                i, j = idx // k, idx % k
                print(f'     Real component ({i},{j}): error = {err[idx]}')
            else:
                i, j = (idx - k*k) // k, (idx - k*k) % k
                print(f'     Imag component ({i},{j}): error = {err[idx]}')
    
    real_part = expectations[:k*k].reshape(k, k)
    imag_part = expectations[k*k:].reshape(k, k)
    return np.complex128(real_part + 1j*imag_part)