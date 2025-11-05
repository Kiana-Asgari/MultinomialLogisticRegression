import numpy as np

from scipy.linalg import sqrtm

from multinomial_logistic.utils import batched_inv
from multinomial_logistic.evaluation.utils import plot_distribution, custom_linspace
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import esd_empirical
from multinomial_logistic.ESD.Marchenko_Pastur_FP_GPU import _complex_integration_gpu




"""
Solving the equation: E_\nu [(I + D\bar S)^{-1} - zI]^{-1} = 
                                                        1/ alpha * bar S
For bar S \in {R}^{k * k},
 nu = Law(Prox(g + Sy; S), g_0), 
       where (g,g_0) ~ N(0, R)
       S, R are the solution of the fixed point equations

D(v,u) = batched_mlogit_jacobian(Prox(g + Sy; S)) 

"""

def recover_density(R_00, alpha, k, k_0, z_imag=1e-3,
                    save_path='multinomial_logistic/data/New_ESD/2_classes'):
    """Recover the spectral density curve across a grid of real parts."""
    z_real_values = custom_linspace(0.002, 0.6, n_points=150)
    density_curve = np.zeros_like(z_real_values)
    last_MP_S = np.complex128(np.eye(k))

    schur, R_01, S, _ = state_evolution_full_recursion(
        R_00=R_00,
        schur_0=R_00,
        R_01_0=np.zeros((k, k)),
        lambda_reg=0,
        alpha=alpha,
        k=k,
        k_0=k_0,
    )
    A = R_01 @ np.linalg.inv(sqrtm(R_00))

    max_density = 0.0
    for idx, z_real in enumerate(z_real_values):
        mp_s, density = stieltjes_inversion(
            R_00,
            schur,
            A,
            S,
            z_real=z_real,
            z_imag=z_imag,
            alpha=alpha,
            k=k,
            k_0=k_0,
            last_MP_S=last_MP_S,
        )
        density_curve[idx] = density
        last_MP_S = mp_s
        max_density = max(max_density, density)
        if max_density > 0 and density < 5e-4 * max_density:
            break

    _, empirical_density = esd_empirical(
        alpha=alpha, k=k, lambda_reg=0, R_00=R_00, max_iter=100, d=250
    )

    plot_distribution(
        density=density_curve,
        empirical_density=empirical_density,
        z_values=z_real_values,
        title=f"ESD theoretical, number of class={k+1:d},R_00={R_00}, alpha={alpha}",
        x_label="eigenvalue",
        y_label="density",
        name=f"ESD_theoretical_nclass={k+1:d}_R_00={R_00}_alpha={alpha}",
        save_path=save_path,
    )

    return density_curve


def stieltjes_inversion(R_00, schur, A, S, z_real, alpha, k, k_0, z_imag,
                        max_iter=350, last_MP_S=None):
    np.random.seed(42)
    MP_S, stieltjes_transform = MP_iteration(
        R_00=R_00,
        schur=schur,
        A=A,
        S=S,
        z_real=z_real,
        z_imag=z_imag,
        alpha=alpha,
        k=k,
        k_0=k_0,
        last_MP_S=last_MP_S,
        max_iter=max_iter,
    )
    density = stieltjes_transform.imag / np.pi
    return MP_S, density


def MP_iteration(R_00, schur, A, S, z_real, z_imag, alpha, k, k_0,
                 last_MP_S=None, tol=1e-3, max_iter=350):
    if last_MP_S is None:
        MP_S_current_inv = np.complex128(np.eye(k))
    else:
        MP_S_current_inv = batched_inv(last_MP_S)
    err = 0

    for t in range(max_iter):  
        MP_S_next_inv = alpha * _MP_F_equation(
            R_00=R_00,
            schur=schur,
            A=A,
            S=S,
            z_real=z_real,
            z_imag=z_imag,
            MP_S_inv=MP_S_current_inv,
            alpha=alpha,
            k=k,
            k_0=k_0,
        )
        MP_S_next = batched_inv(MP_S_next_inv)
        MP_S_current = batched_inv(MP_S_current_inv)
        stieltjes_transform_current =  1/k *alpha* np.trace(MP_S_current)
        stieltjes_transform_next = 1/k *alpha* np.trace(MP_S_next)

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
    return MP_S_current, stieltjes_transform



def _MP_F_equation( R_00, schur, A, S, z_real, MP_S_inv, alpha, k, k_0, z_imag):
    """
    Returns F(S;\nu) = {E_\nu [(I + D*MP_S)^{-1}*D - z_MP I]}^{-1} 
    """
    z = np.complex128(z_real + z_imag*1j)

    expectation = _complex_integration_gpu(
        R_00=R_00,
        schur=schur,
        A=A,
        S=S,
        MP_S_inv=MP_S_inv,
        alpha=alpha,
        k=k,
        k_0=k_0,
    )
    return expectation - z * np.eye(k)