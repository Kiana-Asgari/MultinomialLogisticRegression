import numpy as np

from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline


def phase_transition(R_00, k, k_0, seed=58, n_trials=100, max_alpha=4):
    print('empirical phase transition')

    alpha = max_alpha
    time_step = 0.05 

    for i in range(n_trials):
        alpha = alpha - time_step 
        _, avg_norm, _, _ = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=0,\
                                              R_00=R_00, n_trials=10, d=250)
        print("alpha: ", alpha, "norm: ", avg_norm)
        #if np.linalg.norm(avg_Theta_hat) > 2*k:
        #    time_step = np.max([time_step / 2, 1e-2])
        #    print("reduced time_step: ", time_step)

        if avg_norm> 1e3:
            print('divergence detected at alpha: ', alpha)
            break

    return alpha
