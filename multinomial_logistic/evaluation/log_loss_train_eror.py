import math

import torch

from state_evolution.utils import sphere_mesh_integration, get_primary_device


def train_error(R_00, schur, R_01, S, alpha, k, k_0, seed=42):
    device = get_primary_device()
    dtype = torch.float32

    R_00_tensor = torch.tensor(R_00, device=device, dtype=dtype)
    schur_tensor = torch.tensor(schur, device=device, dtype=dtype)
    R_01_tensor = torch.tensor(R_01, device=device, dtype=dtype)
    S_tensor = torch.tensor(S, device=device, dtype=dtype)

    R00_sqrt = _matrix_sqrt(R_00_tensor)
    R00_sqrt_inv = torch.linalg.inv(R00_sqrt)
    schur_root = _matrix_sqrt(schur_tensor)
    A_tensor = torch.matmul(R_01_tensor, R00_sqrt_inv)
    A_full = torch.matmul(A_tensor, R00_sqrt_inv)
    cov_inv = torch.linalg.inv(schur_tensor)

    y_basis = torch.cat(
        [
            torch.zeros((1, k), dtype=dtype, device=device),
            torch.eye(k, dtype=dtype, device=device),
        ],
        dim=0,
    )

    loss_tensor = sphere_mesh_integration(
        _train_log_loss_integrand,
        S_tensor,
        R00_sqrt,
        schur_root,
        cov_inv,
        A_full,
        A_tensor,
        y_basis,
        k,
        k_0,
        input_dim=k + k_0,
        output_dim=1,
        seed=seed,
        n_radius=16,
        n_polar=7,
        radius=4.5,
        batch_size=200_000,
    )

    loss_value = loss_tensor[0].item()
    print("     train loss: ", loss_value)
    return loss_value


def _train_log_loss_integrand(
    Z_batch,
    S_t,
    R00_sqrt,
    schur_root,
    cov_inv,
    A_full,
    A_t,
    y_basis,
    k,
    k_0,
):
    g_batch, g_0_batch = _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0)
    pdf = _standard_normal_pdf(Z_batch)
    prob_y_batch = _batched_mlogit(g_0_batch)

    J = _batched_mlogit_jacobian(g_batch)
    identity = torch.eye(k, dtype=g_batch.dtype, device=g_batch.device).unsqueeze(0)
    system = identity + torch.einsum("ij,njk->nik", S_t, J)
    det_system = torch.linalg.det(system)

    gradient_batch = _batched_mlogit(g_batch)[:, :-1]
    T_prox_batch = _batched_mult(S_t, gradient_batch) + g_batch
    mean_prox_batch = _batched_mult(A_full, g_0_batch)
    diff = g_batch - mean_prox_batch
    gaussian_IS_weight_batch = torch.einsum("ni,ij,nj->n", diff, cov_inv, diff)

    batch_size = Z_batch.shape[0]
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

    prob_y_reordered = torch.cat(
        [prob_y_batch[:, -1:].T, prob_y_batch[:, :-1].T],
        dim=0,
    )

    logsum = _logsumexp_with_zero(g_batch)
    dot = torch.matmul(y_basis, g_batch.T)
    logloss = logsum.unsqueeze(0) - dot

    weighted = logloss * prob_y_reordered * prox_density
    loss = det_system * weighted.sum(dim=0) * pdf

    return loss.unsqueeze(1)


@torch.no_grad()
def _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0):
    Z_top = Z_batch[:, :k]
    Z_bottom = Z_batch[:, -k_0:]
    g = torch.matmul(Z_bottom, A_t.T) + torch.matmul(Z_top, schur_root.T)
    g_0 = torch.matmul(Z_bottom, R00_sqrt.T)
    return g, g_0


@torch.no_grad()
def _matrix_sqrt(matrix):
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
    return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)


@torch.no_grad()
def _batched_mlogit(beta):
    zeros = beta.new_zeros((beta.shape[0], 1))
    logits = torch.cat([beta, zeros], dim=1)
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def _batched_mlogit_jacobian(beta):
    probabilities = _batched_mlogit(beta)[:, :-1]
    outer_products = torch.einsum("ni,nj->nij", probabilities, probabilities)
    diagonals = torch.zeros_like(outer_products)
    diag_index = torch.arange(probabilities.shape[1], device=beta.device)
    diagonals[:, diag_index, diag_index] = probabilities
    return diagonals - outer_products


@torch.no_grad()
def _batched_mult(matrix, batch):
    return torch.matmul(batch, matrix.T)


@torch.no_grad()
def _standard_normal_pdf(samples):
    d = samples.shape[-1]
    norm_sq = (samples ** 2).sum(dim=-1)
    coeff = (2 * math.pi) ** (-0.5 * d)
    return coeff * torch.exp(-0.5 * norm_sq)


@torch.no_grad()
def _logsumexp_with_zero(beta):
    zeros = beta.new_zeros((beta.shape[0], 1))
    logits = torch.cat([zeros, beta], dim=1)
    return torch.logsumexp(logits, dim=1)


@torch.no_grad()
def _prox_density(
    g_0_batch,
    g_batch,
    y_batch,
    A_full,
    cov_inv,
    S,
    T,
    mean,
    gaussian_IS_weight,
):
    gaussian_point_batch = T - _batched_mult(S, y_batch)
    tilted_gaussian_point_batch = gaussian_point_batch - mean
    exponent = -0.5 * (
        torch.einsum("ni,ij,nj->n", tilted_gaussian_point_batch, cov_inv, tilted_gaussian_point_batch)
        - gaussian_IS_weight
    )
    return torch.exp(exponent)