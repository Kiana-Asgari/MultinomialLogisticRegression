
import numpy as np
from scipy.linalg import sqrtm

from state_evolution.recursion_parts.R_01_recursion import R_01_recursion
from state_evolution.recursion_parts.S_recursion import S_recursion
from state_evolution.recursion_parts.full_R_recursion import R_recursion
div_prox = False



def state_evolution_full_recursion(R_00: np.ndarray,
                                    schur_0: np.ndarray,
                                    R_01_0: np.ndarray,
                                    lambda_reg: float,
                                    alpha: float,
                                    k: int,
                                    k_0: int,
                                    S_0=None,
                                    tol: float=1e-5,
                                    max_iter: int=300,
                                    seed: int=42):
    # AMP state evolution initialization
    np.random.seed(seed)
    
    R_00_sqrtm_inv, R_01_t, schur_t, S_t, divergence, errors = _initialize_state_evolution(R_00, schur_0, R_01_0, lambda_reg,
                                                                                           alpha, k, k_0, S_0=S_0, max_iter=max_iter, seed=seed)
    div_tol = np.linalg.norm(R_00)* 5 * 1e4
    integral_mesh_size = 10
    integral_size = 5.5

    for t in range(max_iter):
        # Step 1: Compute the next state variables
        S_next, schur_next, R_01_next = _compute_next_state_variables(S_t, R_00, schur_t, R_01_t, lambda_reg, alpha, k, k_0, R_00_sqrtm_inv, 
                                                     integral_mesh_size=integral_mesh_size, integral_size=integral_size)    
        errors[t] = np.array([np.linalg.norm(R_01_next - R_01_t), np.linalg.norm(schur_next - schur_t), np.linalg.norm(S_next - S_t)])

        # Step 2: Print the statistics
        _print_stats(t, alpha, errors, S_next, R_01_next, schur_next)

        # Step 3: Check for divergence
        #divergence = check_for_divergence(S_next, S_t, schur_next, schur_t, R_01_next, R_01_t, div_tol, t+1, errors)

            #return schur_t, R_01_t, S_t, divergence
        if all(errors[t] < 1e-4):
            integral_mesh_size = 15
            integral_size = 5.5
        if all(errors[t]<tol) :
            divergence = False
            break

        # Step 4: Update the state variables
        schur_t, R_01_t, S_t = schur_next, R_01_next, S_next

    # Final step: Print the final statistics
  
    return schur_t, R_01_t, S_t, divergence

import time

def _compute_next_state_variables(S_t, R_00, schur_t, R_01_t, 
                                 lambda_reg, alpha, k, k_0,
                                R_00_sqrtm_inv, integral_mesh_size=10, integral_size=5.5):

    S_next = S_recursion(S_t=S_t, R_00=R_00, schur_t=schur_t, R_01_t=R_01_t, 
                         lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, 
                         R_00_sqrtm_inv=R_00_sqrtm_inv,
                          integral_mesh_size=integral_mesh_size, integral_size=integral_size)

    R_01_next, schur_next = R_recursion(S_t=S_t, S_next=S_next, R_00=R_00, schur_t=schur_t, R_01_t=R_01_t, 
                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, 
                                        R_00_sqrtm_inv=R_00_sqrtm_inv,
                                         integral_mesh_size=integral_mesh_size, integral_size=integral_size)

    # R_01_next = R_01_recursion(S_t=S_t, S_next=S_next, R_00=R_00, schur_t=schur_t, R_01_t=R_01_t, 
    #                            lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0, R_00_sqrtm_inv=R_00_sqrtm_inv)

    print('\n','-'*40,'\n')
    return S_next, schur_next, R_01_next

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


def _initialize_state_evolution(R_00: np.ndarray,
                                schur_0: np.ndarray,
                                R_01_0: np.ndarray,
                                lambda_reg: float,
                                alpha: float,
                                k: int,
                                k_0: int,
                                S_0: np.ndarray=None,
                                max_iter: int=300,
                                seed: int=42):
    np.random.seed(seed)
    div_tol = np.linalg.norm(R_00)* 5 * 1e4
    print('*************state evolution iteration started*************')
    print(f'     [initial parameters] lambda: {lambda_reg}', f'alpha: {alpha}', f'k: {k}','R_00: ', R_00)

    R_00_sqrtm_inv = np.linalg.inv(sqrtm(R_00))
    R_01_t = np.zeros_like(R_00)
    schur_t = R_00
    S_t =  np.eye(k)
    divergence = False
    errors = np.zeros((max_iter, 3))
    return R_00_sqrtm_inv, R_01_t, schur_t, S_t, divergence, errors


def _print_stats(t, alpha, errors, S, R_01, schur):             
    print(f'state evolution iteration {t+1} Done (alpha: {alpha})')  
    print(f'     ** R_01 RESIDUAL IS {errors[t,0]}**')
    print(f'     ** SCHUR RESIDUAL IS {errors[t,1]}**')
    print(f'     ** S RESIDUAL IS {errors[t,2]}**\n')

