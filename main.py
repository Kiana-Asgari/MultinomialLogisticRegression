from multinomial_logistic.test import test_integration_1, test_integration_2, test_prox, test_fp_integrands, test_fp_system, test_fp_solver
from multinomial_logistic.test import test_fp_solver
from multinomial_logistic.utils import batched_mlogit_jacobian, batched_mlogit
import numpy as np

from state_evolution.S_fp import S_fp_equation, S_fp_solver_new
from state_evolution.full_recursion import S_recursion, R_01_recursion, schur_recursion
from state_evolution.state_evolution_iteration import state_evolution_fixed_point
from state_evolution.full_recursion import state_evolution_full_recursion
if __name__ == "__main__":
    print("Running main")
    #test_integration_1()
    #test_integration_2()
    #test_prox()
    #test_fp_integrands()
    #test_fp_system()
    k = 2
    k_0 = 2
    R_00 = np.array(np.array([[2,1],
                             [1,2]]))
    #state_evolution_fixed_point(R_00=np.eye(k_0), schur_0=np.eye(k_0), R_01_0=np.zeros((k_0,k)),\
    #                             alpha=30, k=k, k_0=k_0)
    state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                    lambda_reg=0, alpha=10, k=k, k_0=k_0)
    #S_fp_solver(R_00=np.eye(k_0), schur_t=np.eye(k_0), A_t=np.zeros((k_0,k)),\
     #           alpha=10, k=k, k_0=k_0)