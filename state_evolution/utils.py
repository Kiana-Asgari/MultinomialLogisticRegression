import numpy as np
import torch
from cubature import cubature


def integration(integrand, S, R_00, schur_t, A_t, alpha, k, k_0, R_01_t=None, seed=42, fdim=None):
    # Set numpy random seed before cubature call
    np.random.seed(seed)
    
    if fdim is None:
        fdim = k*k
    ndim = k+k_0
    if R_01_t is not None:
        args = (S, R_00, schur_t, A_t, alpha, k, k_0, R_01_t)
    else:
        args = (S, R_00, schur_t, A_t, alpha, k, k_0)
    expectations, err = cubature(integrand, 
                                   args=args, 
                                   ndim=ndim,
                                   vectorized=True,
                                   fdim=fdim,
                                   xmin=[-5]*ndim, 
                                   xmax=[5]*ndim, 
                                   relerr=1e-5,
                                   abserr=1e-6,
                                   maxEval=3_000_000, 
                                   norm=1)

    if np.max(err) > 1e-4:
        print('     **Error in integration is too large**', np.max(err))
    
    if fdim == k*k:
        return expectations.reshape(k, k)
    else: # reshape into two matrices of size k*k and k*k_0
        R_01 = expectations[:k*k].reshape(k, k)
        schur = expectations[k*k:].reshape(k, k)
        return R_01, schur



def mesh_integration(integrand, S, R_00, schur_t, A_t, alpha, k, k_0, R_01_t=None, seed=42, fdim=None, n_mesh=10, size=5.5):
    torch.manual_seed(seed)
    if fdim is None:
        fdim = k * k
    ndim = k + k_0

    dtype = getattr(S, "dtype", torch.float64)
    device = getattr(S, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    S_tensor = torch.as_tensor(S, dtype=dtype, device=device)
    R_00_tensor = torch.as_tensor(R_00, dtype=dtype, device=device)
    schur_tensor = torch.as_tensor(schur_t, dtype=dtype, device=device)
    A_tensor = torch.as_tensor(A_t, dtype=dtype, device=device)
    alpha_tensor = torch.as_tensor(alpha, dtype=dtype, device=device)

    if R_01_t is not None:
        R_01_tensor = torch.as_tensor(R_01_t, dtype=dtype, device=device)
        args = (S_tensor, R_00_tensor, schur_tensor, A_tensor, alpha_tensor, k, k_0, R_01_tensor)
    else:
        args = (S_tensor, R_00_tensor, schur_tensor, A_tensor, alpha_tensor, k, k_0)

    dx = 2.0 * size / n_mesh
    axis = torch.linspace(-size + dx / 2.0, size - dx / 2.0, n_mesh, device=device, dtype=dtype)
    grids = torch.meshgrid(*([axis] * ndim), indexing='ij')
    points = torch.stack([grid.reshape(-1) for grid in grids], dim=-1)
    n_points = points.shape[0]

    batch_size = 50000
    n_batches = (n_points + batch_size - 1) // batch_size

    expectations = torch.zeros(fdim, dtype=dtype, device=device)
    print(f'  --n_batches: {n_batches}, n_points: {n_points}')

    for batch_index in range(n_batches):
        start_idx = batch_index * batch_size
        end_idx = min((batch_index + 1) * batch_size, n_points)
        batch_points = points[start_idx:end_idx]
        batch_result = integrand(batch_points, *args)
        expectations += batch_result.sum(dim=0)

    volume_element = dx ** ndim
    expectations = expectations * volume_element

    if fdim == k * k:
        return expectations.reshape(k, k)
    R_01 = expectations[: k * k].reshape(k, k)
    schur = expectations[k * k :].reshape(k, k)
    return R_01, schur