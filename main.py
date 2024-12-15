from multinomial_logistic.test import test_integration_1, test_integration_2, test_prox, test_fp_integrands, test_fp_system, test_fp_solver
from multinomial_logistic.test import test_fp_solver
from multinomial_logistic.utils import batched_mlogit_jacobian, batched_mlogit
import numpy as np
from state_evolution.S_fp import S_fp_integrand, S_fp_equation, S_fp_solver

from state_evolution.S_fp import S_fp_equation, S_fp_solver

if __name__ == "__main__":
    print("Running main")
    #test_integration_1()
    #test_integration_2()
    #test_prox()
    #test_fp_integrands()
    #test_fp_system()
    print(S_fp_solver( R_00=np.eye(2), schur_root_t=np.eye(2), A_t=np.eye(2), alpha=10, k=2, k_0=2))
