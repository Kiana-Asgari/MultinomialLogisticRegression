import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

from multinomial_logistic.utils import mlogit, batched_mlogit, log_sum_exp_batch
from cubature import cubature
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_scalar_mult, batched_normal_basis
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.utils import batched_sqrtm 
from multinomial_logistic.evaluation.utils import plot_array
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline

def plot_test_error_vs_lambda( R_00, lambda_reg_min, lambda_reg_max \
                             ,k, k_0, max_iter, save_path):
    print('plotting test error vs alpha... for parameters:')
    print('     R_00 = ', R_00)

    alpha_values = [2,4,6,10]               
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, max_iter, endpoint=False)

    irr_error = irreducible_error(R_00, k, k_0, alpha=None)
    test_error_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    test_error_empirical_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    legends = ['alpha=2', 'alpha=4', 'alpha=6', 'alpha=10']

    for i, alpha in enumerate(alpha_values):
        for j, lambda_reg in enumerate(lambda_reg_values):
            schur, R_01, S, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                            lambda_reg=lambda_reg/2, alpha=alpha, k=k, k_0=k_0)
            if divergence:
                print('     **Divergence detected**')
                break

            print(f'     schur = {schur}')
            print(f'     R_01 = {R_01}')
            print(f'     S = {S}')
            test_error_batches[i,j] = test_error(R_00, schur, R_01, k, k_0, alpha)
            _, _, test_error_empirical,_ = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=lambda_reg,\
                                                    d=350, R_00=R_00, n_trials=200)
            test_error_empirical_batches[i,j] = test_error_empirical
            print(f' alpha = {alpha:.2f}   test_error_batches = {test_error_batches[i,j]}', 'empirical = ', test_error_empirical_batches[i,j])
        

    title = f"log loss train error vs lambda_reg, number of class={k+1:d},R_00= ({R_00})"
    name = f"test_error_vs_lambda_reg_nclass={k+1:d}_R_00={R_00}" 
    plot_array(lambda_reg_values, test_error_batches, test_error_empirical_batches,\
                legends=legends, irreducible_error=irr_error,\
                title=title,\
                x_label="lambda_reg", y_label="train log loss",\
                name=name, save_path=save_path)
    
    print('irreducible error: ', irr_error)


def plot_test_error_vs_alpha( R_00, alpha_min, alpha_max \
                             ,k, k_0, max_iter, save_path):
    print('plotting test error vs alpha... for parameters:')
    print('     R_00 = ', R_00)

    R_00_values = [ np.eye(k), R_00]               
    alpha_values = np.linspace(alpha_min, alpha_max, max_iter, endpoint=False)

    irr_error2 = irreducible_error(R_00, k, k_0, alpha=None)
    irr_error1 = irreducible_error(np.eye(k), k, k_0, alpha=None)
    test_error_batches = np.zeros((len(R_00_values), len(alpha_values)))
    test_error_empirical_batches = np.zeros((len(R_00_values), len(alpha_values)))

    legends = ['R_00=I_k', f'R_00={R_00.flatten()}']


    for i, R_00 in enumerate(R_00_values):
        schur_0=R_00
        R_01_0=np.zeros((k_0,k))
        S_0=np.eye(k)
        for j, alpha in enumerate(alpha_values):
            schur_0, R_01_0, S_0, divergence = state_evolution_full_recursion(R_00=R_00, schur_0=schur_0, R_01_0=R_01_0, S_0=S_0,\
                                            lambda_reg=0, alpha=alpha, k=k, k_0=k_0)
            if divergence:
                print('     **Divergence detected**')
                break

            print(f'     schur = {schur_0}')
            print(f'     R_01 = {R_01_0}')
            print(f'     S = {S_0}')
            test_error_batches[i,j] = test_error(R_00, schur_0, R_01_0, k, k_0, alpha)
            _, _, test_error_empirical,_ = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=0,\
                                                    d=250, R_00=R_00, n_trials=200)
            test_error_empirical_batches[i,j] = test_error_empirical
            print(f' R_00 = {R_00}   test_error_batches = {test_error_batches[i,j]}', 'empirical = ', test_error_empirical_batches[i,j])
        

    title = f"log loss train error vs alpha, number of class={k+1:d})"
    name = f"test_error_vs_alpha_nclass={k+1:d}" 
    plot_array(alpha_values, test_error_batches, test_error_empirical_batches,\
                legends=legends, irreducible_error=[irr_error1, irr_error2],\
                title=title,\
                x_label="alpha", y_label="test log loss",\
                multiple_irreducible_error=True,\
                name=name, save_path=save_path)
    
    print('irreducible error: ', irr_error1, irr_error2)
    



####################################################################################################

def test_error( R_00, schur, R_01, k, k_0, alpha,seed=42):
    np.random.seed(seed)
    A = R_01 @ sqrtm(np.linalg.inv(R_00))
    #loss = integration(_test_error_integrand, R_00, schur, A, alpha, k, k_0)
    loss_mesh = mesh_integration(_test_error_integrand, R_00, schur, A, alpha, k, k_0)
    print('     test loss: ', loss_mesh)
    return loss_mesh



def irreducible_error(R_00, k, k_0, alpha, seed=42):
    np.random.seed(seed)
    A = sqrtm(R_00)
    schur = np.zeros((k,k))
    irreducible_error = integration(_test_error_integrand, R_00, schur, A, alpha, k, k_0)
    print('     done computing irreducible error: ', irreducible_error)
    return irreducible_error














####################################################################################################

def _test_error_integrand(Z_batch, R_00, schur, A, alpha, k, k_0,):
    #print('     computing test error integrand...')
    #print('     Z_batch shape:', Z_batch.shape)
    N = Z_batch.shape[0]
    schur_root_t = sqrtm(schur)
    g_1_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root_t  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)

    # Compute the loss
    prob_y_batch = batched_mlogit(g_0_batch)

    loss = np.zeros(N)
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        logloss_batch = log_sum_exp_batch(g_1_batch) - np.einsum('ij,ij->i', y_batch, g_1_batch)

        loss += logloss_batch * prob_y_batch[:, i] # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    return loss * pdf



####################################################################################################

def mesh_integration(integrand, R_00, schur, A, alpha, k, k_0, seed=42, size=5, n_mesh=15):
    # Set numpy random seed before mesh integration
    np.random.seed(seed)
    fdim = 1
    ndim = int(k+k_0)
    # Create mesh grid for ndim dimensions using midpoint rule
    # Divide [-size, size] into n_mesh intervals, sample at midpoints
    dx = 2 * size / n_mesh
    axes = [np.linspace(-size + dx/2, size - dx/2, n_mesh) for _ in range(ndim)]
    grids = np.meshgrid(*axes, indexing='ij')
    
    # Flatten the grids to get all points: shape (n_mesh^ndim, ndim)
    points = np.stack([grid.flatten() for grid in grids], axis=-1)
    n_points = points.shape[0]
    
    # Prepare args for integrand
    args = (R_00, schur, A, alpha, k, k_0)
    
    # Compute integration using batches for vectorized computation
    batch_size = 100000
    n_batches = (n_points + batch_size - 1) // batch_size
    
    expectations = np.zeros(fdim)
    print(f'  --n_batches: {n_batches}, n_points: {n_points}')
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_points)
        batch_points = points[start_idx:end_idx] # Shape: (batch_size, ndim)
        # Call integrand with batched points
        batch_result = integrand(batch_points, *args)  # Shape: (batch_size, fdim)       
        # Sum over the batch
        expectations += np.sum(batch_result, axis=0)
    
    # Compute volume element (dx^ndim for midpoint rule)
    volume_element = dx ** ndim
    
    # Multiply by volume element to get Riemann sum
    expectations *= volume_element
    return expectations




def integration(integrand, R_00, schur, A, alpha, k, k_0):
    fdim = 1
    ndim = int(k+k_0)
    expectations, err = cubature(integrand, args=( R_00, schur, A, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-5]*ndim, xmax=[5]*ndim, abserr=1e-5, relerr=1e-5,
                                  maxEval= 2_000_000, norm=2)
    if err.item() > 1e-4:
        print('     **[Warning] log loss test error integration error is too large**', err)
    #for e in err:
    #   if e > 1e-3:
    #     print('     **[Warning] state evolution integration error is too large**')
    #     break
    #print('     done integrating')
    return expectations
