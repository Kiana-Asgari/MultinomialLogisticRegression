import numpy as np

from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature
import numpy as np
import casadi as ca



from state_evolution.functions import ODE_batched
from multinomial_logistic.utils import batched_mlogit, batched_outer, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import coloring_transform, batched_mult
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system
from multinomial_logistic.utils import wrapper, batched_mlogit_jacobian, batched_inv
from multinomial_logistic.evaluation.utils import plot_array

from state_evolution.full_recursion import state_evolution_full_recursion

from multinomial_logistic.ESD_theoretical.Marchenko_Pastur_FP import stieltjes_inversion





def recover_density_via_ODE( R_00, alpha, k, k_0, z_imag=1e-3,\
                    save_path='multinomial_logistic/data/ODE_density/2_classes'):
    """
    Recovers the full density indise the support of the ESD 
    from the solution of the fixed point equation
    """
    print(' Starting to recover density with ODE...')

    z_real_values = np.linspace(0.02, 0.5, num=100, endpoint=False)
    density_curve  = np.zeros_like(z_real_values)   
    legends = ['density']
    max_density = 0
    last_MP_S = np.complex128(np.eye(k))

    schur, R_01, S, _ = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k,k)),\
                                            lambda_reg=0, alpha=alpha, k=k, k_0=k)
    A = R_01 @ np.linalg.inv(sqrtm(R_00))

    S_MP_0, density_0 = initial_point(R_00, schur, A, S, z_real=z_real_values[0], z_imag=z_imag,\
                                       alpha=alpha, k=k, k_0=k_0, last_MP_S=last_MP_S)
    density_curve[0] = density_0
    print('initial point done with density:', density_0,' at z=', z_real_values[0])

    S_final = solve_implicit_ode(k, ode_implicit_function, S_MP_0, z_real_values, R_00, schur, A, alpha, k_0)
    print('ODE solution done', S_final)






###################################################################################
# 1) ODE implicit function
###################################################################################
def ode_implicit_function(S_MP, dS_MP , R_00, schur, A, S, alpha, k, k_0):
    expectation = complex_integration(_ode_integrand,S_MP, dS_MP , R_00, schur, A, S, alpha, k, k_0)
    #return np.linalg.inv(expectation - z * np.eye(k))
    equation = -dS_MP + alpha * S_MP @ (expectation - np.eye(k)) @ S_MP
    return equation






###################################################################################
def initial_point(R_00, schur, A, S, z_real, z_imag, alpha, k, k_0, last_MP_S):
    MP_S, density = stieltjes_inversion(R_00, schur, A, S, z_real=z_real, z_imag=z_imag,\
                                alpha=alpha, k=k, k_0=k_0, last_MP_S=last_MP_S)
    return MP_S, density

def _ode_integrand(G_batch, S_MP, dS_MP , R_00, schur, A, S, alpha, k, k_0):
    N = G_batch.shape[0]
    schur_root = sqrtm(schur)

    # Coloring transform
    g_batch, g_0_batch = coloring_transform(G_batch, A=A, R_00=R_00, schur_root=schur_root  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(G_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k), dtype=np.complex128)

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S) # prox(g + yS; S)
        MP_batch = ODE_batched(V_batch=prox_g_batch, S_MP=S_MP, dS_MP=dS_MP, k=k) # N*k*k (I_k + S @ Jp(prox(g + yS; S)))^{-1} @ Jp(prox(g + yS; S))
        integrand += batched_scalar_mult(MP_batch, prob_y_batch[:, i])    

    integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)
   

    integrand_flat = integrand.reshape(N, -1) #flattened, N*k^2 

    # to deal with complex integration, we split the real and imaginary parts of the integrand
    return np.hstack([integrand_flat.real, integrand_flat.imag])  # N*2k^2




#####################################################################################




def complex_integration(integrand, S_MP, dS_MP , R_00, schur, A, S, alpha, k, k_0):
    fdim = 2*k*k
    ndim = k+k_0

    expectations, err = cubature(integrand, args=(S_MP, dS_MP , R_00, schur, A, S, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim=fdim,
                                  xmin=[-3.4]*ndim, xmax=[3.4]*ndim, 
                                  abserr=1e-5,
                                  maxEval=200_000, norm=1)
    
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


###################################################3
def define_symbols(k):
    """
    Define the symbolic variables for the ODE system.
    :param k: Size of the complex matrix S (k x k)
    :return: CasADi symbolic variables S, dS/dz
    """
    S_real = ca.MX.sym("S_real", k, k)  # Real part of S
    S_imag = ca.MX.sym("S_imag", k, k)  # Imaginary part of S
    dS_real = ca.MX.sym("dS_real", k, k)  # Real part of dS/dz
    dS_imag = ca.MX.sym("dS_imag", k, k)  # Imaginary part of dS/dz

    # Create a complex representation using a custom class or keep real/imag separate
    # Option 1: Return separate real and imaginary parts
    return S_real, S_imag, dS_real, dS_imag

def wrap_F(F_inference, R_00, schur, A, alpha, k, k_0):
    """
    Wrap the black-box function F(S, dS/dz) for CasADi compatibility.
    :param F_inference: Function that computes F(S, dS/dz, ...)
    :param R_00, schur, A, alpha, k, k_0: Additional arguments for F
    :return: Callable F(S, dS/dz)
    """
    def F(S, dS_dz):
        S_val = S.full()  # Evaluate S
        dS_val = dS_dz.full()  # Evaluate dS/dz
        return ca.DM(F_inference(S, dS_dz, R_00, schur, A, S_val, dS_val, alpha, k, k_0))  # Return as CasADi DM object

    return F

def solve_implicit_ode(k, F_inference, S0, z_span, R_00, schur, A, alpha, k_0):
    """
    Solve the implicit ODE F(S, dS/dz) = 0.
    :param k: Size of the complex matrix S (k x k)
    :param F_inference: Black-box function for F(S, dS/dz, ...)
    :param S0: Initial condition (k x k complex matrix)
    :param z_span: Integration range [start, end]
    :param R_00, schur, A, alpha, k_0: Additional arguments for F
    :return: Final solution S(z)
    """
    # Define symbolic variables
    S_real, S_imag, dS_real, dS_imag = define_symbols(k)


    # Wrap the black-box function F
    def F_wrapped(S_real, S_imag, dS_real, dS_imag):
        return F_inference(S_real, S_imag, dS_real, dS_imag, R_00, schur, A, S, alpha, k, k_0)

    # Define the ODE residual: F(S, dS/dz) = 0
    residual = F_wrapped(S_real, S_imag, dS_real, dS_imag)
    residual_real = ca.vertcat(residual.real().reshape((-1, 1)), residual.imag().reshape((-1, 1)))

    # Define CasADi's implicit solver (e.g., IDAS)
    solver = ca.integrator(
        "solver",
        "idas",
        {
            "x": ca.vertcat(S_real, S_imag),  # State variables (real + imaginary parts of S)
            "z": ca.vertcat(dS_real, dS_imag),  # Derivatives (real + imaginary parts of dS/dz)
            "ode": residual_real,  # Flattened residual
        },
        {"tf": z_span[1] - z_span[0]},  # Final time (difference of z_span)
    )

    # Flatten initial condition into real and imaginary parts
    S0_real = S0.real.flatten()
    S0_imag = S0.imag.flatten()

    # Solve the implicit ODE
    solution = solver(x0=np.hstack([S0_real, S0_imag]))
    S_solution = solution["xf"]

    # Extract the final solution and reshape into a complex matrix
    S_real_sol = np.array(S_solution[:k**2]).reshape((k, k))
    S_imag_sol = np.array(S_solution[k**2:]).reshape((k, k))
    S_final = S_real_sol + 1j * S_imag_sol

    return S_final