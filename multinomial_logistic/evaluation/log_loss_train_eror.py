import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from multinomial_logistic.utils import mlogit, batched_mlogit, log_sum_exp_batch
from cubature import cubature
from multinomial_logistic.integration import coloring_transform
from multinomial_logistic.utils import batched_mult, batched_outer, batched_scalar_mult, batched_normal_basis
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.evaluation.utils import plot_array
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline


    





def train_error(R_00, schur, R_01, S,alpha, k, k_0, seed=42):

    np.random.seed(seed)
    #loss = integrate(_train_log_loss_integrand, R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=alpha, k=k, k_0=k_0)
    loss = mesh_integration(_train_log_loss_integrand, R_00=R_00, schur=schur, R_01=R_01, S=S, alpha=alpha, k=k, k_0=k_0)
    print('     train loss: ', loss)
    return loss



#########################
# Log loss integrand
#########################
def _train_log_loss_integrand(Z_batch, R_00, schur, R_01, S, alpha, k, k_0):
    # Compute schur complement
    N = Z_batch.shape[0]
    R_00_inv = np.linalg.inv(R_00)
    schur_root = sqrtm(schur)
    A = R_01 @ sqrtm(R_00_inv)
    
    # Coloring transform
    g_batch, g_0_batch = coloring_transform(Z_batch, A=A, R_00=R_00, schur_root=schur_root  , alpha=alpha, k=k, k_0=k_0) # (g,g_0) ~ N(0, R)
    prob_y_batch = batched_mlogit(g_0_batch)
    
    loss = np.zeros(N)
    for i in range(-1, k):
        y_batch = batched_normal_basis(i, k, N) # Y = (0,1,0...0) batch
        prox_g_batch, div_prox = prox_fp_iteration(g_batch + batched_mult(S, y_batch), S) # prox(g + yS; S)

        
        if div_prox:
            print('     **prox Divergence detected**')
            break
        logloss_batch = log_sum_exp_batch(prox_g_batch) - np.einsum('ij,ij->i', y_batch, prox_g_batch)
        loss += logloss_batch * prob_y_batch[:, i] # (I + S @ Jp(prox(g + yS; S)))^{-1} * p(y)  

    pdf = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0)).pdf(Z_batch)
    return loss * pdf
    
   
    



def integrate(integrand, R_00, schur, R_01, S, alpha, k, k_0):
    #print('     integrating... ')
    fdim = 1
    ndim = k+k_0
    expectations, err = cubature(integrand, args=(R_00, schur, R_01, S, alpha, k, k_0,), ndim=ndim,
                                  vectorized=True,
                                  fdim= fdim ,xmin=[-3.6]*ndim, xmax=[3.6]*ndim, abserr = 1e-5,
                                  maxEval=1_500_000, norm=2)
    if err.item() > 1e-4:
        print('     **[Warning] train error integration error is too large**, err=', err)
    #print('     done integrating')
    return expectations     






def mesh_integration(integrand, R_00, schur, R_01, S, alpha, k, k_0, seed=42, size=4.5, n_mesh=12):
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
    args = (R_00, schur, R_01, S, alpha, k, k_0)
    
    # Compute integration using batches for vectorized computation
    batch_size = 50000
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