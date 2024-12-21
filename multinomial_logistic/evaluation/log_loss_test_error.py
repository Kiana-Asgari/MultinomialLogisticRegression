import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

from multinomial_logistic.utils import mlogit, batched_mlogit, log_sum_exp_batch
from cubature import cubature
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_scalar_mult, batched_normal_basis
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.utils import batched_sqrtm 
from multinomial_logistic.evaluation.utils import plot_array




def plot_test_error_vs_lambda_reg(mean_0, variance_0, R_00, lambda_reg_min, lambda_reg_max, alpha,\
                              k, k_0, max_iter, save_path):
    print('plotting test error vs lambda_reg... for parameters:')
    print('     mean_0 = ', mean_0)
    print('     variance_0 = ', variance_0)
    print('     R_00 = ', R_00)

    irr_error = irreducible_error(mean_0, variance_0, k, k_0)
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, max_iter, endpoint=False)
    test_error_values = []

    for lambda_reg in lambda_reg_values:

        schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
        if divergence:
            print('     **Divergence detected**')
            break
        print(f'     schur = {schur}')
        print(f'     R_01 = {R_01}')
        print(f'     S = {S}')
        test_error_values.append(test_error(mean_0, variance_0, R_00, schur, R_01, k, k_0))
        

    title = f"log loss test error vs lambda_reg for alpha={alpha:.2f}, number of class={k+1:d}, \mu_0 = N({mean_0},{variance_0})"
    name = f"test_error_vs_lambda_reg_alpha={alpha:.2f}_nclass={k+1:d}_mu_0={mean_0}_var_0={variance_0}" 
    plot_array(lambda_reg_values, test_error_values, irr_error, title=title,\
                x_label="lambda_reg", y_label="test log loss",\
                name=name, save_path=save_path)


def plot_test_error_vs_alpha(mean_0, variance_0, R_00, alpha_min, alpha_max \
                             ,k, k_0, max_iter, save_path):
    print('plotting test error vs alpha... for parameters:')
    print('     mean_0 = ', mean_0)
    print('     variance_0 = ', variance_0)
    print('     R_00 = ', R_00)
    lambda_reg_values = [0, 0.01]               
    alpha_values = np.linspace(alpha_min, alpha_max, max_iter, endpoint=False)

    irr_error = irreducible_error(mean_0, variance_0, k, k_0)
    test_error_batches = np.zeros((len(lambda_reg_values), len(alpha_values)))
    legends = ['lambda_reg=0', 'lambda_reg=0.01']

    for i, lambda_reg in enumerate(lambda_reg_values):
        for j, alpha in enumerate(alpha_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                            lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
            if divergence:
                print('     **Divergence detected**')
                break

            print(f'     schur = {schur}')
            print(f'     R_01 = {R_01}')
            print(f'     S = {S}')
            test_error_batches[i,j] = test_error(mean_0, variance_0, R_00, schur, R_01, k, k_0)
        print(f' lambda_reg = {lambda_reg:.2f}   test_error_batches = {test_error_batches[i,:]}')
        

    title = f"log loss train error vs alpha, number of class={k+1:d},R_00= ({R_00})"
    name = f"test_error_vs_alpha_nclass={k+1:d}_R_00={R_00}" 
    plot_array(alpha_values, test_error_batches, legends=legends, irreducible_error=irr_error,\
                title=title,\
                x_label="alpha", y_label="train log loss",\
                name=name, save_path=save_path)
    



####################################################################################################

def test_error(mean_0, variance_0, R_00, schur, R_01, k, k_0):
    print('     computing test error... with parameters:')

    loss_mc, err = monte_carlo_expectation(_test_error_integrand, mean_0=mean_0, variance_0=variance_0, \
                                           R_00=R_00, schur=schur, R_01=R_01, k=k, k_0=k_0, \
                                           dim=2*(k+k_0), monte_carlo=True)
    print('     done computing test error with loss_mc: ', loss_mc)
    return loss_mc



def irreducible_error(mean_0, variance_0, k, k_0):
    irreducible_error_mc, err = monte_carlo_expectation(_irreducible_error_integrand, mean_0=mean_0,\
                                                         variance_0=variance_0, R_00=0, schur=0, R_01=0,\
                                                         k=k, k_0=k_0, dim=k_0+(k+k_0), monte_carlo=True)
    print('     done computing irreducible error: ', irreducible_error_mc)
    return irreducible_error_mc














####################################################################################################\\
def _irreducible_error_integrand(Z_batch, mean_0, variance_0, R_00, schur, R_01, k, k_0, monte_carlo):
    N = Z_batch.shape[0]
    Z_theta = Z_batch[:,:k_0] # (N, k_0), for recovering theta_0
    Z_g = Z_batch[:,-(k_0+k):] # (N, k+k_0), for recovering g,g_0

    # recover theta_0 from Z_theta~ N(0, I)
    theta_0_batch = np.tile(mean_0, (N, 1)) + batched_mult(sqrtm(variance_0), Z_theta[:,:k_0])


    g_1_batch, g_0_batch = _recover_g(Z_g, theta_0_batch, theta_0_batch, k, k_0) # (g,g_0) ~ N(0, R)

    # Compute the loss
    prob_y_batch = batched_mlogit(g_0_batch)

    loss = log_sum_exp_batch(g_1_batch) 
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        loss -= (np.einsum('ij,ij->i', y_batch, g_1_batch) * prob_y_batch[:, i])

    if not monte_carlo:
        pdf = multivariate_normal(mean=np.zeros(k+(k+k_0)), cov=np.eye(k+(k+k_0))).pdf(Z_batch)
    else:
        pdf = 1

    return loss * pdf



def _test_error_integrand(Z_batch, mean_0, variance_0, R_00, schur, R_01, k, k_0, monte_carlo):
    #print('     computing test error integrand...')
    #print('     Z_batch shape:', Z_batch.shape)
    N = Z_batch.shape[0]
    Z_theta = Z_batch[:,:k+k_0] # (N, k+k_0), for recovering theta_1, theta_0
    Z_g = Z_batch[:,-(k_0+k):] # (N, k+k_0), for recovering g,g_0

    # recover theta_1, X @ theta_1, X @ theta_0
    theta_1_batch, theta_0_batch = _recover_theta(Z_theta, mean_0, variance_0, R_00, schur, R_01, k, k_0)
    g_1_batch, g_0_batch = _recover_g(Z_g, theta_1_batch, theta_0_batch, k, k_0) # (g,g_0) ~ N(0, R)

    # Compute the loss
    prob_y_batch = batched_mlogit(g_0_batch)

    loss = np.zeros(N)
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        logloss_batch = log_sum_exp_batch(g_1_batch) - np.einsum('ij,ij->i', y_batch, g_1_batch)

        loss += logloss_batch * prob_y_batch[:, i] # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    if not monte_carlo:
        pdf = multivariate_normal(mean=np.zeros(2*(k+k_0)), cov=np.eye(2*(k+k_0))).pdf(Z_batch)
    else:
        pdf = 1

    return loss * pdf



####################################################################################################
def _recover_theta(Z_theta, mean_0, variance_0, R_00, schur, R_01, k, k_0):
    N = Z_theta.shape[0]

    Z_theta_0 = Z_theta[:,:k_0]
    Z_theta_1 = Z_theta[:,k:]

    theta_0_batch = np.tile(mean_0, (N, 1)) + batched_mult(sqrtm(variance_0), Z_theta_0)

    schur_root = sqrtm(schur)
    A = R_01 @ np.linalg.inv(R_00)
    mean = np.einsum('ij,nj->ni', A, theta_0_batch)
    theta_1_batch = mean + np.einsum('ij,nj->ni', schur_root, Z_theta_1)

    #print('     done recovering theta... with parameters:', theta_1_batch, theta_0_batch)
    return theta_1_batch, theta_0_batch


def _recover_g(Z_g, theta_1_batch, theta_0_batch, k, k_0):
    theta_batch = np.concatenate([theta_0_batch, theta_1_batch], axis=1)

    covariance_0 = batched_sqrtm(np.einsum('ni,nj->nij', theta_batch, theta_batch)) # R = [R_00, R_01; R_10, R_11]
    g_batch = np.einsum('nij,nj->ni', covariance_0, Z_g)

    g_0_batch = g_batch[:,:k_0]
    g_1_batch = g_batch[:,-k:]
    #print('     done recovering g... with parameters:', g_1_batch, g_0_batch)

    return g_1_batch, g_0_batch





def monte_carlo_expectation(func, mean_0, variance_0, R_00, schur, R_01, k, k_0, dim, \
                            initial_samples=10_000, max_samples=10_000_000, 
                            tol=2 * 1e-4, monte_carlo=True):

    # Dimension of the standard normal vector
    # First Run of the monte carlo
    
    # Generate batch of standard normal vectors
    monte_carlo = True
    
    # Start with an initial number of samples
    num_samples = initial_samples
    
    # A variable to keep track of the previous estimate for error checking
    prev_estimate = None
    current_estimate = None
    err = np.inf

    while err > tol and num_samples <= max_samples:
        # Generate batch of standard normal vectors
        # Half with np.random.normal, half with scipy for consistency with original approach
        half_samples = num_samples // 2
        
        z_batch = np.random.normal(0, 1, (half_samples, dim))
        func_values_1 = func(z_batch, mean_0=mean_0, variance_0=variance_0, \
                                           R_00=R_00, schur=schur, R_01=R_01, k=k, k_0=k_0, monte_carlo=True)
        expectation_1 = np.mean(func_values_1, axis=0)

        # Second batch
        z_batch = multivariate_normal.rvs(mean=np.zeros(dim), cov=np.eye(dim), size=half_samples)
        func_values_2 = func(z_batch, mean_0=mean_0, variance_0=variance_0, \
                                           R_00=R_00, schur=schur, R_01=R_01, k=k, k_0=k_0, monte_carlo=True)
        expectation_2 = np.mean(func_values_2, axis=0)

        # Combined estimate
        current_estimate = (expectation_1 + expectation_2) / 2

        # Compute error if we have a previous estimate
        if prev_estimate is not None:
            # Simple difference norm as a stopping criterion
            err = np.linalg.norm(prev_estimate - current_estimate)
        else:
            # If no previous estimate, just set error high and proceed
            err = np.inf
        
        # Print warnings if necessary
        if err > tol:
            # Increase samples and try again
            prev_estimate = current_estimate
            num_samples = int(2*num_samples)
        else:
            # We are done
            break

        if num_samples > max_samples:
            print('     **[Warning] Reached maximum number of samples before meeting tolerance**')
            break

    if err > tol:
        print('     **[Warning] Adaptive sampling error is still too large**')
    
    return expectation_2, err





