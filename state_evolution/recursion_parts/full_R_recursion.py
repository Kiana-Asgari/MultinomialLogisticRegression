import torch
import math
from state_evolution.functions import _matrix_sqrt, _matrix_inv
from state_evolution.utils import sphere_mesh_integration



def R_recursion(
    S_t_tensor,
    S_next_tensor,
    R_00_tensor,
    schur_tensor,
    R_01_tensor,
    lambda_tensor,
    alpha_tensor,
    k,
    k_0,
    R_00_sqrtm_inv_tensor=None,
    integral_mesh_size=10,
    integral_size=5.5,
):
    device = S_t_tensor.device
    dtype = S_t_tensor.dtype 

    if R_00_sqrtm_inv_tensor is None:
        R_00_sqrt = _matrix_sqrt(R_00_tensor)
        A_tensor = torch.matmul(R_01_tensor, _matrix_inv(R_00_sqrt))
    else:
        A_tensor = torch.matmul(R_01_tensor, R_00_sqrtm_inv_tensor)
    y_basis = torch.cat(
        [
            torch.zeros((1, k), dtype=dtype, device=device),
            torch.eye(k, dtype=dtype, device=device),
        ],
        dim=0,
    )

    combined_integrand = sphere_mesh_integration(
        _R_integrand_with_prox_density_fully_vectorized,
        S_t_tensor,
        R_00_tensor,
        schur_tensor,
        A_tensor,
        y_basis,
        alpha_tensor,
        k,
        k_0,
        input_dim=k + k_0,
        output_dim=2 * k * k,
        batch_size=400_000,
        n_radius=32, #14 is good enough I think. Below is bad
        n_polar=integral_mesh_size,
        radius=integral_size,
        dtype=dtype
    )
    R_01_integrand = combined_integrand[: k * k].reshape(k, k)
    schur_integrand = combined_integrand[k * k :].reshape(k, k)

    schur = alpha_tensor * torch.matmul(
        S_next_tensor,
        torch.matmul(schur_integrand, S_next_tensor),
    )
    identity_k = torch.eye(k, dtype=dtype, device=device)
    R_01 = (
        torch.matmul(identity_k - 2.0 * alpha_tensor * lambda_tensor * S_next_tensor, R_01_tensor)
        - alpha_tensor * torch.matmul(S_next_tensor, R_01_integrand)
    )

    return R_01, schur


def _R_integrand_with_prox_density_fully_vectorized(
    Z_batch,
    S_t,
    R_00,
    schur_t,
    A_t,
    y_basis,
    alpha,
    k,
    k_0,
):
    batch_size = Z_batch.shape[0]

    schur_root = _matrix_sqrt(schur_t)
    R00_sqrt = _matrix_sqrt(R_00)
    A_full = torch.matmul(A_t, _matrix_inv(R00_sqrt))
    cov_inv = _matrix_inv(schur_t)

    g_batch, g_0_batch = _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0)
    pdf = _standard_normal_pdf(Z_batch)
    prob_y_batch = _batched_mlogit(g_0_batch)
    score_jacobian_batch = _score_jacobian_inverse(g_batch, S_t, k)
    det_score_jacobian_batch = torch.reciprocal(torch.linalg.det(score_jacobian_batch))

    gradient_batch = _batched_mlogit(g_batch)[:, :-1]
    T_prox_batch = _batched_mult(S_t, gradient_batch) + g_batch
    mean_prox_batch = _batched_mult(A_full, g_0_batch)
    diff = g_batch - mean_prox_batch
    gaussian_IS_weight_batch = torch.einsum("ni,ij,nj->n", diff, cov_inv, diff)

    
    repeats = y_basis.shape[0]
    y_flat = y_basis.repeat_interleave(batch_size, dim=0)
    g_flat = g_batch.repeat(repeats, 1)
    g_0_flat = g_0_batch.repeat(repeats, 1)
    T_flat = T_prox_batch.repeat(repeats, 1)
    mean_flat = mean_prox_batch.repeat(repeats, 1)
    gaussian_IS_flat = gaussian_IS_weight_batch.repeat(repeats)

    prox_density_flat = _prox_density(
        g_0_batch=g_0_flat,
        g_batch=g_flat,
        y_batch=y_flat,
        A_full=A_full,
        cov_inv=cov_inv,
        S=S_t,
        T=T_flat,
        mean=mean_flat,
        gaussian_IS_weight=gaussian_IS_flat,
    )
    score_flat = _score_batched(g_flat, y_flat)

    prox_density_reshaped = prox_density_flat.view(repeats, batch_size)
    score_reshaped = score_flat.view(repeats, batch_size, k)

    prob_y_reordered = torch.cat(
        [prob_y_batch[:, -1:].T, prob_y_batch[:, :-1].T],
        dim=0,
    )
    weights = (
        prob_y_reordered
        * prox_density_reshaped
        * det_score_jacobian_batch.unsqueeze(0)
    )

    schur_integrand = torch.einsum("in,inj,ink->njk", weights, score_reshaped, score_reshaped)
    R_01_integrand = torch.einsum("in,inj,nk->njk", weights, score_reshaped, g_0_batch)

    R_01_integrand = _batched_scalar_mult(R_01_integrand, pdf)
    schur_integrand = _batched_scalar_mult(schur_integrand, pdf)

    R_01_flat = R_01_integrand.reshape(batch_size, -1)
    schur_flat = schur_integrand.reshape(batch_size, -1)
    return torch.cat([R_01_flat, schur_flat], dim=1)





def _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0):
    Z_top = Z_batch[:, :k]
    Z_bottom = Z_batch[:, -k_0:]
    g = torch.matmul(Z_bottom, A_t.T) + torch.matmul(Z_top, schur_root.T)
    g_0 = torch.matmul(Z_bottom, R00_sqrt.T)
    return g, g_0


import math
def _standard_normal_pdf(samples: torch.Tensor)-> torch.Tensor:
    d = samples.size(-1)
    norm_sq = (samples ** 2).sum(dim=-1)
    coeff = (2 * math.pi) ** (-0.5 * d)
    return coeff * torch.exp(-0.5 * norm_sq)


def _batched_mult(matrix, batch):
    return torch.matmul(batch, matrix.T)


def _batched_scalar_mult(tensor, weights):
    return tensor * weights.view(-1, 1, 1)


def _batched_mlogit(beta):
    zeros = torch.zeros((beta.shape[0], 1), dtype=beta.dtype, device=beta.device)
    logits = torch.cat([beta, zeros], dim=1)
    return torch.softmax(logits, dim=1)


def _batched_mlogit_jacobian(beta):
    probabilities = _batched_mlogit(beta)[:, :-1]
    outer_products = torch.einsum("ni,nj->nij", probabilities, probabilities)
    diagonals = torch.zeros_like(outer_products)
    diag_index = torch.arange(probabilities.shape[1], device=beta.device)
    diagonals[:, diag_index, diag_index] = probabilities
    return diagonals - outer_products


def _score_batched(V_batch, y_batch):
    return _batched_mlogit(V_batch)[:, :-1] - y_batch


def _score_jacobian_inverse(V_batch, S, k):
    J = _batched_mlogit_jacobian(V_batch)
    identity = torch.eye(k, dtype=V_batch.dtype, device=V_batch.device).unsqueeze(0)
    S_term = torch.einsum("ij,njk->nik", S, J)
    system = identity + S_term
    return torch.linalg.inv(system)


def _prox_density(
    g_0_batch,
    g_batch,
    y_batch,
    A_full,
    cov_inv,
    S,
    T=None,
    mean=None,
    gaussian_IS_weight=None,
):
    if T is None:
        gradient_batch = _batched_mlogit(g_batch)[:, :-1]
        gaussian_point_batch = _batched_mult(S, gradient_batch) + g_batch - _batched_mult(S, y_batch)
    else:
        gaussian_point_batch = T - _batched_mult(S, y_batch)

    if mean is None:
        mean = _batched_mult(A_full, g_0_batch)

    tilted_gaussian_point_batch = gaussian_point_batch - mean

    if gaussian_IS_weight is None:
        gaussian_IS_weight = torch.einsum("ni,ij,nj->n", g_batch - mean, cov_inv, g_batch - mean)

    exponent = -0.5 * (
        torch.einsum(
            "ni,ij,nj->n",
            tilted_gaussian_point_batch,
            cov_inv,
            tilted_gaussian_point_batch,
        )
        - gaussian_IS_weight
    )
    return torch.exp(exponent)

