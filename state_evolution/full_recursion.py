
import numpy as np

from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature


from state_evolution.functions import score_batched, score_jacobian_batched
from multinomial_logistic.utils import batched_mlogit, batched_outer, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import coloring_transform, batched_mult
from multinomial_logistic.prox import prox_fp_iteration, ProximalOperatorError
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system
from multinomial_logistic.utils import wrapper


"""
    State Evoltion Recursion for the Regularized Multinomial Logistic Regression.
    The state evolution recursion is used to compute the fixed point of the state evolution equations.
    The state evolution equations are:

        S_{t+1} = 1/alpha * ((I - E[(I + S @ Jp(prox(g + yS; S)))^{-1}] + 2*lambda_reg*S_t))^{-1} @ S_t
        R_01_{t+1} = (I - alpha*2*lambda_reg * S_{t+1}) @ R_01_t - alpha * S_{t+1} @ E[(p(prox(g + yS; S)) - y)g_0.T] 
        schur_{t+1} = alpha * S_{t+1} @ E[(p(v)-y) @ (p(v)-y).T] @ S_{t+1}

    Note that the prox is computed at the S_t.
    This recursion is used to compute the fixed point of the state evolution equations
    
        (1/alpha - 1) * I + (2*lambda_reg) * S =  E[(I + S @ Jp(v))^{-1}] 
        (-2*lambda_reg) @ R_01 = E[(p(v) - y)g_0.T] 
        schur = alpha * S @ E[(p(v)-y) @ (p(v)-y).T] @ S

"""

div_prox = False


def state_evolution_full_recursion(R_00, schur_0, R_01_0, lambda_reg, alpha, k, k_0,\
                                    S_0=None,tol=1e-5, max_iter=300, seed=42):
    np.random.seed(seed)


    div_tol = np.linalg.norm(R_00)* 5 * 1e4
    print('*************state evolution iteration started*************')
    print(f'     [initial parameters] lambda: {lambda_reg}', f'alpha: {alpha}', f'k: {k}','R_00: ', R_00)

    R_00_sqrtm_inv = np.linalg.inv(sqrtm(R_00))
    R_01_t = R_01_0
    schur_t = schur_0
    if S_0 is None:
        S_t =  np.eye(k)
    else:
        S_t = S_0
    divergence = False
    errors = np.zeros((max_iter, 3))
    
    for t in range(max_iter):
        print(f'     state evolution iteration {t+1} started for alpha: {alpha}... ')
        S_next = S_recursion(S_t=S_t, R_00=R_00, schur_t=schur_t, R_01_t=R_01_t, 
                             lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, R_00_sqrtm_inv=R_00_sqrtm_inv)
        schur_next = schur_recursion(S_t=S_t, S_next=S_next, R_00=R_00, schur_t=schur_t, R_01_t=R_01_t, 
                                     lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, R_00_sqrtm_inv=R_00_sqrtm_inv)
        R_01_next = R_01_recursion(S_t=S_t, S_next=S_next, R_00=R_00, schur_t=schur_t, R_01_t=R_01_t, 
                                   lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, R_00_sqrtm_inv=R_00_sqrtm_inv)
        errors[t] = np.array([np.linalg.norm(R_01_next - R_01_t), np.linalg.norm(schur_next - schur_t), np.linalg.norm(S_next - S_t)])

            

        print(f'     **THE R_01 RESIDUAL IS {errors[t,0]}**')
        print(f'     **THE SCHUR RESIDUAL IS {errors[t,1]}**')
        print(f'     **THE S RESIDUAL IS {errors[t,2]}**')
        print(f'     **THE S norm IS {np.linalg.norm(S_next)}**')

        divergence = check_for_divergence(S_next, S_t, schur_next, schur_t, \
                                          R_01_next, R_01_t, div_tol, t+1, errors)
        if divergence:
            print(' diverged for values: schur_t: ', schur_t, 'R_01_t: ', R_01_t, 'S_t: ', S_t)
            #return schur_t, R_01_t, S_t, divergence


        if all(errors[t]<tol) :
            divergence = False
            break

        schur_t, R_01_t, S_t = schur_next, R_01_next, S_next

    if divergence:
        print('     **Divergence detected due to reaching max_iter**')
    print(f'     **done with state evolution recursion. schur_t: {schur_t}, R_01_t: {R_01_t}, S_t: {S_t}, R_00: {R_00}, alpha: {alpha}, k: {k}, k_0: {k_0}**')
    return schur_t, R_01_t, S_t, divergence





def S_recursion(S_t, R_00, schur_t, R_01_t, lambda_reg, alpha, k, k_0, R_00_sqrtm_inv=None):
    if R_00_sqrtm_inv is None:
        A_t = R_01_t @ np.linalg.inv(sqrtm(R_00))
    else:
        A_t = R_01_t @ R_00_sqrtm_inv
    integrand = integration(_S_fp_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    S = 1/alpha * np.linalg.inv(np.eye(k) - integrand + 2*lambda_reg*S_t) @ S_t
    return S

def R_01_recursion(S_t, S_next, R_00, schur_t, R_01_t, lambda_reg, alpha, k, k_0, R_00_sqrtm_inv=None):
    if R_00_sqrtm_inv is None:
        A_t = R_01_t @ np.linalg.inv(sqrtm(R_00))
    else:
        A_t = R_01_t @ R_00_sqrtm_inv
    integrand = integration(_R_01_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    R_01 =  (np.eye(k) - alpha*2*lambda_reg * S_next) @ R_01_t \
            - alpha * S_next @ integrand 
    return R_01

def schur_recursion(S_t, S_next, R_00, schur_t, R_01_t, lambda_reg, alpha, k, k_0, R_00_sqrtm_inv=None):
    if R_00_sqrtm_inv is None:
        A_t = R_01_t @ np.linalg.inv(sqrtm(R_00))
    else:
        A_t = R_01_t @ R_00_sqrtm_inv
    integrand = integration(_schur_integrand, S_t, R_00, schur_t, A_t, alpha, k, k_0)
    schur = alpha * S_next @ integrand @ S_next
    return schur

#####################################################################################

def _schur_integrand(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0, monte_carlo=False):
    """
    full equation: 1/alpha * S_t E[score(v_t, y_t)@ score(v_t, y_t).T] S_t = (R/R_00)_{t+1}
    integrand: score(v_t, y_t)@ score(v_t, y_t).T | g,g_0 ~ N(0, R_t)
    score(v_t, y _t) = P(v) - y
    v = prox(g + Sy)
    """
    global div_prox 
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur_t)
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch, div_prox = prox_fp_iteration(g_batch + batched_mult(S_t, y_batch), S_t) # prox(g + yS; S)
        score_batch = score_batched(prox_g_batch, y_batch) # p(prox(g + yS; S)) - y
        integrand += batched_scalar_mult(batched_outer(score_batch, score_batch), prob_y_batch[:, i]) # score @ score.T   

    if not monte_carlo:
        integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened



def _R_01_integrand(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0, monte_carlo=False):   
    """
    R_{01, t+1} = R_{01,t} - alpha * E[score(v_t, y_t)@ G_0.T]
    score(v_t, y _t) = P(v) - y
    v = prox(g + Sy)
    """
    global div_prox 
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur_t)   
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch, div_prox = prox_fp_iteration(g_batch + batched_mult(S_t, y_batch), S_t) # prox(g + yS; S)
        score_batch = score_batched(prox_g_batch, y_batch) # p(prox(g + yS; S)) - y
        integrand += batched_scalar_mult(batched_outer(score_batch, g_0_batch), prob_y_batch[:, i]) # score @ g_0.T

    if not monte_carlo:
        integrand = batched_scalar_mult(integrand, pdf) 

    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened







def _S_fp_integrand(Z_batch, S_t, R_00, schur_t, A_t, alpha, k, k_0, monte_carlo=False):
    N = Z_batch.shape[0]
    schur_root = sqrtm(schur_t)
    global div_prox 
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root, alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y_batch = batched_mlogit(g_0_batch)

    integrand = np.zeros((N, k, k))

    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batched
        prox_g_batch, div_prox = prox_fp_iteration(g_batch + batched_mult(S_t, y_batch), S=S_t) # prox(g + yS; S)
        score_jacobian_batch = score_jacobian_batched(prox_g_batch, S_t, k) # (I + S @ Jp(prox(g + yS; S)))^{-1}
    
        #print('score_jacobian for y = ', i)
        #print('    .... with eigenvalues: ', np.linalg.eigvals(score_jacobian_batch[0]))
        integrand += batched_scalar_mult(score_jacobian_batch, prob_y_batch[:, i]) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  
   
    if not monte_carlo:
        integrand = batched_scalar_mult(integrand, pdf) # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y) * p(g,g_0)
    
    if k == 1:
        return integrand.reshape(-1)#flattened  
    return integrand.reshape(N, -1) #flattened


#####################################################################################

def integration(integrand, S, R_00, schur_t, A_t, alpha, k, k_0, seed=42):
    # Set numpy random seed before cubature call
    np.random.seed(seed)
    
    fdim = k*k
    ndim = k+k_0
    expectations, err = cubature(integrand, 
                                   args=(S, R_00, schur_t, A_t, alpha, k, k_0,), 
                                   ndim=ndim,
                                   vectorized=True,
                                   fdim=fdim,
                                   xmin=[-5]*ndim, 
                                   xmax=[5]*ndim, 
                                   abserr=1e-5,
                                   relerr=1e-5,
                                   maxEval=1_000_000, 
                                   norm=1)

    if np.max(err) > 1e-4:
        print('     **Error in integration is too large**', np.max(err))

    return expectations.reshape(k, k)




###############################################################


def check_for_divergence(S_next, S_t, schur_next, schur_t, R_01_next, R_01_t,\
                          div_tol, iter,errors):
    global div_prox
    divergence = False

    if div_prox:
        print("[DIVERGENCE] Halting state evolution due to [prox] divergence")
        divergence = True

    for i in range(1, iter):
        if all(errors[i,j] - errors[i-1,j] > 1e-5 for j in range(3)): #change
            print("[DIVERGENCE] Halting state evolution due to [all errors increase > 0]")
            print(f"Error jump detected: {errors[i] - errors[i-1]}")
            divergence = True
            break

        if any(errors[i,j] > div_tol/2 for j in range(3)):
            print("[DIVERGENCE] Halting state evolution due to one [error] too large")
            divergence = True
            break

        if any(errors[i,j] - errors[i-1,j] > 0.5 and errors[i,j] > 5 \
               for j in range(3)):
            print("[DIVERGENCE] Halting state evolution due to one [error] too large and growing")
            divergence = True
            break


 

    if np.linalg.norm(S_next) > div_tol or np.linalg.norm(schur_next) > div_tol or np.linalg.norm(R_01_next) > div_tol:
        print("[DIVERGENCE] Halting state evolution due to [norm] divergence")
        divergence = True
    return divergence 
