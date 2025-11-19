import math
import numpy as np
import torch

from multinomial_logistic.utils import batched_inv
from multinomial_logistic.evaluation.utils import plot_distribution, custom_linspace
from state_evolution.full_recursion import state_evolution_full_recursion
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import esd_empirical
from state_evolution.utils import sphere_mesh_integration




def stieltjes_inversion(R_00, schur, A, S, z_real, alpha, k, k_0, z_imag,
                        tol=1e-4, max_iter=350, last_MP_S=None):
    np.random.seed(42)

    MP_S, stieltjes_transform = MP_iteration(R_00=R_00, schur=schur, A=A, S=S, z_real=z_real,
                                            tol=tol, max_iter=max_iter,
                                          last_MP_S=last_MP_S, z_imag=z_imag, alpha=alpha, k=k, k_0=k_0)
    density = stieltjes_transform.imag / np.pi
    print('     density at z_real = ', z_real + z_imag*1j, ' R_00 = ', R_00.flatten(), '     density = ', density)
    return MP_S, density
    



def MP_iteration(R_00, schur, A, S, z_real, z_imag, alpha, k, k_0, \
              last_MP_S=None, tol=1e-3, max_iter=350):
    """
    Solves the fixed point equation: E_\nu [(I + D\bar S)^{-1} - z_MP I]^{-1} = 1/ alpha * \bar S
    """
    if last_MP_S is None:
        MP_S_current_inv = np.complex128(np.eye(k))
    else:
        MP_S_current_inv = batched_inv(last_MP_S)
    err = 0

    print(f'*********[Starting MP iteration]... for z_real = {z_real+z_imag*1j}, starting MP_S_inv = {MP_S_current_inv.flatten()}')
    
    for t in range(max_iter):  
        MP_S_next_inv =  alpha * _MP_F_equation(R_00=R_00, schur=schur, A=A, S=S,\
                                            z_real=z_real, z_imag=z_imag, MP_S_inv=MP_S_current_inv, alpha=alpha, k=k, k_0=k_0)
        MP_S_next = batched_inv(MP_S_next_inv)
        MP_S_current = batched_inv(MP_S_current_inv)
        stieltjes_transform_next = 1/k *alpha* np.trace(MP_S_next)

        err_image = np.linalg.norm(MP_S_next.imag - MP_S_current.imag)/np.linalg.norm(MP_S_next.imag)
        err_real = np.linalg.norm(MP_S_next.real - MP_S_current.real)/np.linalg.norm(MP_S_next.real)
        err = np.max([err_image, err_real])
        print(f'    [iter={t}] err={err:.5e} for z = {z_real+z_imag*1j}')
        MP_S_current_inv = MP_S_next_inv
 
        if stieltjes_transform_next.imag < 0:
            print('     WARN: stieltjes_transform_next.imag < 0')
            break

        if err < tol:
            break


    stieltjes_transform = alpha *1/k * np.trace(MP_S_current)

    print('     stieltjes_transform = ', stieltjes_transform)
    print('                     Err = ', err)
    print('                 MP iteration finished with iteration ', t, '*******************')
    return MP_S_current, stieltjes_transform



def _MP_F_equation( R_00, schur, A, S, z_real, MP_S_inv, alpha, k, k_0, z_imag):
    """
    Returns F(S;\nu) = {E_\nu [(I + D*MP_S)^{-1}*D - z_MP I]}^{-1} 
    """
    z = np.complex128(z_real + z_imag*1j)

    # expectation = _complex_integration_gpu(R_00=R_00, schur=schur, A=A, S=S,
    #                            MP_S_inv=MP_S_inv, alpha=alpha, k=k, k_0=k_0, sphere=False)
    expectation = _complex_integration_gpu(R_00=R_00, schur=schur, A=A, S=S,
                               MP_S_inv=MP_S_inv, alpha=alpha, k=k, k_0=k_0, sphere=True)

    return expectation - z * np.eye(k)




@torch.no_grad()
def _MP_integrand_gpu(Z_batch, S_t, R00_sqrt, schur_root, cov_inv, A_full, A_t, y_basis, MP_S_inv, k, k_0):
    """Torch-based integrand for the Marchenko-Pastur fixed-point equation."""
    batch_size = Z_batch.shape[0]

    g_batch, g_0_batch = _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0)
    pdf = _standard_normal_pdf(Z_batch)
    prob_y_batch = _batched_mlogit(g_0_batch)

    score_jacobian_batch = _score_jacobian_inverse(g_batch, S_t, k)
    det_score_jacobian = torch.linalg.det(score_jacobian_batch)
    volume_factor_batch = torch.reciprocal(det_score_jacobian)

    gradient_batch = _batched_mlogit(g_batch)[:, :-1]
    T_prox_batch = _batched_mult(S_t, gradient_batch) + g_batch
    mean_prox_batch = _batched_mult(A_full, g_0_batch)
    diff = g_batch - mean_prox_batch
    gaussian_IS_weight_batch = torch.einsum("ni,ij,nj->n", diff, cov_inv, diff)

    repeats = y_basis.shape[0]
    y_flat = y_basis.repeat_interleave(batch_size, dim=0)
    g_flat = g_batch.repeat(repeats, 1)
    g0_flat = g_0_batch.repeat(repeats, 1)
    T_flat = T_prox_batch.repeat(repeats, 1)
    mean_flat = mean_prox_batch.repeat(repeats, 1)
    gaussian_IS_flat = gaussian_IS_weight_batch.repeat(repeats)

    prox_density_flat = _prox_density(
        g_0_batch=g0_flat,
        g_batch=g_flat,
        y_batch=y_flat,
        A_full=A_full,
        cov_inv=cov_inv,
        S=S_t,
        T=T_flat,
        mean=mean_flat,
        gaussian_IS_weight=gaussian_IS_flat,
    )
    prox_density = prox_density_flat.view(repeats, batch_size)
    prob_y_reordered = torch.cat([prob_y_batch[:, -1:].T, prob_y_batch[:, :-1].T], dim=0)
    weight_sum = torch.sum(prob_y_reordered * prox_density, dim=0)

    joint_density = weight_sum * volume_factor_batch * pdf
    MP_batch = _MP_batched(g_batch, MP_S_inv)
    integrand = _batched_scalar_mult(MP_batch, joint_density.to(MP_batch.dtype))

    integrand_flat = integrand.reshape(batch_size, -1)
    return torch.cat([integrand_flat.real, integrand_flat.imag], dim=1)

    


#####################################################################################

def _complex_integration_gpu(R_00, schur, A, S, MP_S_inv, alpha, k, k_0, *,
                            size=4.5, batch_size=1_100_000, 
                            seed=42, sphere=False, 
                            n_radius=20, n_polar=7):
    """Compute complex expectations using torch mesh integration."""
    del alpha  # retained for API co    mpatibility

    device = torch.device("cuda")
    real_dtype = torch.float32 #TODO: for 32, batch_size = 400_000, for 64, batch_size = 10_000
    complex_dtype = torch.complex64

    R_00_tensor = torch.tensor(R_00, dtype=real_dtype, device=device)
    schur_tensor = torch.tensor(schur, dtype=real_dtype, device=device)
    A_tensor = torch.tensor(A, dtype=real_dtype, device=device)
    S_tensor = torch.tensor(S, dtype=real_dtype, device=device)

    R00_sqrt = _matrix_sqrt(R_00_tensor)
    R00_sqrt_inv = torch.linalg.inv(R00_sqrt)
    schur_root = _matrix_sqrt(schur_tensor)
    cov_inv = torch.linalg.inv(schur_tensor)
    A_full = torch.matmul(A_tensor, R00_sqrt_inv)

    y_basis = torch.cat([
        torch.zeros((1, k), dtype=real_dtype, device=device),
        torch.eye(k, dtype=real_dtype, device=device),
    ], dim=0)

    MP_S_inv_tensor = torch.tensor(MP_S_inv, dtype=complex_dtype, device=device)

    output_dim = 2 * k * k
    if not sphere:
        pass
    if sphere:
        expectations = sphere_mesh_integration(
                    _MP_integrand_gpu,
                    S_tensor,
                    R00_sqrt,
                    schur_root,
                    cov_inv,
                    A_full,
                    A_tensor,
                    y_basis,
                    MP_S_inv_tensor,
                    k,
                    k_0,
                    input_dim=k + k_0,
                    output_dim=output_dim,
                    seed=seed,
                    radius=size,
                    n_polar=n_polar,
                    n_radius=n_radius,
                    batch_size=batch_size,
                )
    expectations = expectations.to(torch.float64)
    real_part = expectations[:k * k].reshape(k, k)
    imag_part = expectations[k * k:].reshape(k, k)
    complex_matrix = real_part + 1j * imag_part
    return complex_matrix.cpu().numpy().astype(np.complex128)


#####################################################################
# Torch helper utilities
#####################################################################


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    symmetric_matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
    return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)


def _coloring_transform(Z_batch: torch.Tensor, A_t: torch.Tensor, R00_sqrt: torch.Tensor,
                        schur_root: torch.Tensor, k: int, k_0: int):
    Z_top = Z_batch[:, :k]
    Z_bottom = Z_batch[:, -k_0:]
    g = torch.matmul(Z_bottom, A_t.T) + torch.matmul(Z_top, schur_root.T)
    g_0 = torch.matmul(Z_bottom, R00_sqrt.T)
    return g, g_0


def _standard_normal_pdf(samples: torch.Tensor) -> torch.Tensor:
    d = samples.size(-1)
    norm_sq = (samples ** 2).sum(dim=-1)
    coeff = samples.new_tensor((2 * math.pi) ** (-0.5 * d))
    return coeff * torch.exp(-0.5 * norm_sq)


def _batched_mult(matrix: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return torch.matmul(batch, matrix.T)


def _batched_scalar_mult(tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights_view = weights.view(-1, 1, 1)
    if tensor.is_complex():
        weights_view = weights_view.to(tensor.dtype)
    return tensor * weights_view


def _batched_mlogit(beta: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((beta.shape[0], 1), dtype=beta.dtype, device=beta.device)
    logits = torch.cat([beta, zeros], dim=1)
    return torch.softmax(logits, dim=1)


def _batched_mlogit_jacobian(beta: torch.Tensor) -> torch.Tensor:
    probabilities = _batched_mlogit(beta)[:, :-1]
    outer_products = torch.einsum("ni,nj->nij", probabilities, probabilities)
    diagonals = torch.zeros_like(outer_products)
    diag_index = torch.arange(probabilities.shape[1], device=beta.device)
    diagonals[:, diag_index, diag_index] = probabilities
    return diagonals - outer_products


def _score_jacobian_inverse(V_batch: torch.Tensor, S: torch.Tensor, k: int) -> torch.Tensor:
    J = _batched_mlogit_jacobian(V_batch)
    identity = torch.eye(k, dtype=V_batch.dtype, device=V_batch.device).unsqueeze(0)
    system = identity + torch.einsum("ij,njk->nik", S, J)
    return torch.linalg.inv(system)


def _prox_density(*, g_0_batch: torch.Tensor, g_batch: torch.Tensor, y_batch: torch.Tensor,
                  A_full: torch.Tensor, cov_inv: torch.Tensor, S: torch.Tensor,
                  T: torch.Tensor, mean: torch.Tensor, gaussian_IS_weight: torch.Tensor) -> torch.Tensor:
    gaussian_point_batch = T - _batched_mult(S, y_batch)
    tilted_gaussian_point_batch = gaussian_point_batch - mean
    exponent = -0.5 * (
        torch.einsum("ni,ij,nj->n", tilted_gaussian_point_batch, cov_inv, tilted_gaussian_point_batch)
        - gaussian_IS_weight
    )
    return torch.exp(exponent)


def _MP_batched(V_batch: torch.Tensor, MP_S_inv: torch.Tensor) -> torch.Tensor:
    J = _batched_mlogit_jacobian(V_batch).to(MP_S_inv.dtype)
    MP_S_inv_batch = MP_S_inv.unsqueeze(0).expand(V_batch.shape[0], -1, -1)
    inv_term = torch.linalg.inv(MP_S_inv_batch + J)
    score_batch = torch.matmul(MP_S_inv_batch, inv_term)
    return torch.matmul(score_batch, J)