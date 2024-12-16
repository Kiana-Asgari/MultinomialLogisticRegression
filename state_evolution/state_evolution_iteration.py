from state_evolution.R_recursion import R_01_recursion, schur_recursion
from multinomial_logistic.utils import wrapper
from state_evolution.S_fp import S_fp_solver_new
import numpy as np
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system


def state_evolution_iteration(R_00, S_t, schur_t, R_01_t, alpha, k, k_0):
    schur_root_next = schur_recursion(S_t, R_00, schur_t, R_01_t, alpha, k, k_0)
    R_01_next = R_01_recursion(S_t, R_00, schur_t, R_01_t, alpha, k, k_0)
    return schur_root_next, R_01_next

def state_evolution_fixed_point(R_00, schur_0, R_01_0, alpha, k, k_0, tol=1e-4, max_iter=10000):
    print('*************state evolution iteration started*************')
    R_01_t = R_01_0
    schur_t = schur_0
    S_t = np.eye(k)

    for t in range(max_iter):
        print(f'     state evolution iteration {t+1} started... ')
        S_next = S_fp_solver(R_00, schur_t, R_01_t, alpha, k, k_0)
        schur_next, R_01_next = state_evolution_iteration(R_00, S_next, schur_t, R_01_t, alpha, k, k_0)
        print(f'     state evolution iteration {t+1} finished with parameters: R_01={R_01_next.flatten()}, schur={schur_next.flatten()}, S={S_next.flatten()}')
        equations = fixed_point_system(wrapper(S_next, R_01_next @ np.linalg.inv(R_00), schur_next), R_00, alpha, k, k_0)
        print(f'     **THE RESIDUAL IS {equations}**')
        print(f'     **change in parameters R_01, schur, S: {np.linalg.norm(R_01_next - R_01_t)}, {np.linalg.norm(schur_next - schur_t)}, {np.linalg.norm(S_next - S_t)}**')
        if all(np.linalg.norm(next - current) < tol for next, current in [
            (R_01_next, R_01_t), 
            (schur_next, schur_t), 
            (S_next, S_t)
        ]):
            print(f'     state evolution iteration {t+1} finished with parameters: R_01={R_01_next.flatten()}, schur={schur_next.flatten()}, S={S_next.flatten()}')
            print(f'     **THE RESIDUAL IS {equations}**')
            break
        schur_t, R_01_t, S_t = schur_next, R_01_next, S_next
    return schur_t, R_01_t, S_t