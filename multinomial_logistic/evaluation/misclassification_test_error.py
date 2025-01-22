import numpy as np

from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature


from state_evolution.functions import score_batched, score_jacobian_batched
from multinomial_logistic.utils import batched_mlogit, batched_outer, batched_scalar_mult, batched_mult, batched_normal_basis
from multinomial_logistic.integration import coloring_transform, batched_mult
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system
from multinomial_logistic.utils import wrapper
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.evaluation.utils import plot_array



def plot_misclass_test_error_vs_alpha(R_00, alpha_min, alpha_max ,k, k_0, max_iter, save_path):
    print('plotting misclassification test error vs alpha... for parameters:')
    print('     R_00 = ', R_00)
    alpha_values = np.linspace(alpha_min, alpha_max, max_iter, endpoint=False)
    lambda_reg_values = [0]
    legends = ['lambda_reg=0']


    misclass_test_error_batches = np.zeros((len(lambda_reg_values), len(alpha_values)))

    for i, lambda_reg in enumerate(lambda_reg_values):
        for j, alpha in enumerate(alpha_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                            lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
            A_t = R_01 @ np.linalg.inv(R_00)
            misclass_test_error_batches[i, j] = misclassification_test_error(S, R_00, schur, A_t, alpha, k, k_0)
            print('misclass_test_error_batches[i, j]', misclass_test_error_batches[i, j])
        
    title = f"misclassification test error vs alpha, number of class={k+1:d},R_00= ({R_00})"
    name = f"misclassification_test_error_vs_alpha_nclass={k+1:d}_R_00={R_00}" 

    plot_array( x_data=alpha_values, y_data_batch=misclass_test_error_batches, legends=legends,\
                title=title,\
                x_label="alpha", y_label="misclassification test error",\
                name=name, save_path=save_path)





def plot_misclass_test_error_vs_lambda(R_00, lambda_reg_min, lambda_reg_max ,k, k_0, max_iter, save_path):
    print('plotting misclassification test error vs alpha... for parameters:')
    print('     R_00 = ', R_00)
    alpha_values =  [1.5, 2, 4]
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, max_iter, endpoint=False)

    misclass_test_error_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    legends = ['alpha=1.5', 'alpha=2', 'alpha=4']

    for i, alpha in enumerate(alpha_values):
        for j, lambda_reg in enumerate(lambda_reg_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                            lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
            A_t = R_01 @ np.linalg.inv(sqrtm(R_00))
            misclass_test_error_batches[i, j] = misclassification_test_error(S, R_00, schur, A_t, alpha, k, k_0)
            print('misclass_test_error_batches[i, j]', misclass_test_error_batches[i, j])
        
    title = f"misclassification test error vs lambda_reg, number of class={k+1:d},R_00= ({R_00})"
    name = f"misclassification_test_error_vs_lambda_reg_nclass={k+1:d}_R_00={R_00}" 
    plot_array( x_data=2* lambda_reg_values, y_data_batch=misclass_test_error_batches, legends=legends,\
                title=title,\
                x_label="lambda_reg", y_label="misclassification test error",\
                name=name, save_path=save_path)




def misclassification_test_error(S, R_00, schur_t, A_t, alpha, k, k_0, monte_carlo=False):
    accuracy = integration(_misclassification_integrand, S, R_00, schur_t, A_t, alpha, k, k_0)
    #print('integrand.shape', integrand.shape)
    return 1-accuracy


def _misclassification_integrand(Z_batch, S, R_00, schur_t, A_t, alpha, k, k_0):
    #print(' in _misclassification_integrand')
    N = Z_batch.shape[0]
    #print(Z_batch.shape)
    schur_root_t = sqrtm(schur_t)
    g_1_batch, g_0_batch = coloring_transform(Z_batch, A=A_t, R_00=R_00, schur_root=schur_root_t,
                                               alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    prob_y0_batch = batched_mlogit(g_0_batch)
    prob_y1_batch = np.hstack([g_1_batch, np.zeros((g_1_batch.shape[0], 1))])
    y1_batch = np.argmax(prob_y1_batch, axis=1)
    prob_y0_of_y1 = prob_y0_batch[np.arange(N), y1_batch]
    integrand = prob_y0_of_y1 * pdf
    return integrand



############################
def integration(integrand, S, R_00, schur_t, A_t, alpha, k, k_0):
    fdim = 1
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(S, R_00, schur_t, A_t, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-3.5]*ndim, xmax=[3.5]*ndim, abserr=1e-5,
                                  maxEval= 70_000_000, norm=2)
    if err.item() > 1e-4:
        print('     **[Warning] misclassification test error integration error is too large**', err)
    #for e in err:
    #   if e > 1e-3:
    #     print('     **[Warning] state evolution integration error is too large**')
    #     break
    #print('     done integrating')
    return expectations
