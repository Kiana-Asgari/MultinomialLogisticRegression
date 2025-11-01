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







def misclassification_test_error(S, R_00, schur_t, A_t, alpha, k, k_0, monte_carlo=False, seed=42):
    np.random.seed(seed)

    #accuracy = integration(_misclassification_integrand, S, R_00, schur_t, A_t, alpha, k, k_0)
    accuracy_mesh = mesh_integration(_misclassification_integrand, S, R_00, schur_t, A_t, alpha, k, k_0)
    print('     accuracy_mesh: ', accuracy_mesh)
    #print('integrand.shape', integrand.shape)
    return 1-accuracy_mesh








def irreducible_misclassification_error(R_00, k, k_0, alpha):
    A = sqrtm(R_00)
    schur = np.zeros((k,k))
    S=None
    irreducible_error = 1-integration(_misclassification_integrand, S, R_00, schur, A, alpha, k, k_0)
    print('     done computing misclassification irreducible error: ', irreducible_error)
    return irreducible_error






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
                                  fdim= fdim ,xmin=[-3.8]*ndim, xmax=[3.8]*ndim, abserr=1e-5,
                                  maxEval= 2_000_000, norm=2)
    if err.item() > 1e-4:
        print('     **[Warning] misclassification test error integration error is too large**', err)
    #for e in err:
    #   if e > 1e-3:
    #     print('     **[Warning] state evolution integration error is too large**')
    #     break
    #print('     done integrating')
    return expectations


def mesh_integration(integrand, S, R_00, schur_t, A_t, alpha, k, k_0, seed=42, size=6, n_mesh=20):
    # Set numpy random seed before mesh integration
    np.random.seed(seed)
    fdim = 1
    ndim = k+k_0
    # Create mesh grid for ndim dimensions using midpoint rule
    # Divide [-size, size] into n_mesh intervals, sample at midpoints
    dx = 2 * size / n_mesh
    axes = [np.linspace(-size + dx/2, size - dx/2, n_mesh) for _ in range(ndim)]
    grids = np.meshgrid(*axes, indexing='ij')
    
    # Flatten the grids to get all points: shape (n_mesh^ndim, ndim)
    points = np.stack([grid.flatten() for grid in grids], axis=-1)
    n_points = points.shape[0]
    
    # Prepare args for integrand
    args = (S, R_00, schur_t, A_t, alpha, k, k_0)
    
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