import math

import torch

from state_evolution.utils import sphere_mesh_integration, get_primary_device


def test_error(R_00, schur, R_01, k, k_0, alpha, seed=42):

    device = get_primary_device()
    dtype = torch.float32

    R_00_tensor = torch.tensor(R_00, device=device, dtype=dtype)
    schur_tensor = torch.tensor(schur, device=device, dtype=dtype)
    R_01_tensor = torch.tensor(R_01, device=device, dtype=dtype)

    R00_sqrt = _matrix_sqrt(R_00_tensor)
    R00_sqrt_inv = torch.linalg.inv(R00_sqrt)
    schur_root = _matrix_sqrt(schur_tensor)
    A_tensor = torch.matmul(R_01_tensor, R00_sqrt_inv)

    y_basis = torch.cat(
        [
            torch.zeros((1, k), dtype=dtype, device=device),
            torch.eye(k, dtype=dtype, device=device),
        ],
        dim=0,
    )

    loss_tensor = sphere_mesh_integration(
        _test_error_integrand,
        schur_root,
        A_tensor,
        R00_sqrt,
        y_basis,
        k,
        k_0,
        input_dim=k + k_0,
        output_dim=1,
        seed=seed,
        n_radius=18,
        n_polar=12,
       # n_azimuth=10,
        radius=5,
        batch_size=450_000,
    )

    loss_value = loss_tensor[0].item()
    print("     test loss: ", loss_value)
    return loss_value


def _test_error_integrand(
    Z_batch,
    schur_root,
    A_t,
    R00_sqrt,
    y_basis,
    k,
    k_0,
):
    g_1_batch, g_0_batch = _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0)
    pdf = _standard_normal_pdf(Z_batch)
    prob_y_batch = _batched_mlogit(g_0_batch)

    logsum = _logsumexp_with_zero(g_1_batch)
    dot = torch.matmul(y_basis, g_1_batch.T)
    logloss = logsum.unsqueeze(0) - dot

    prob_y_reordered = torch.cat(
        [prob_y_batch[:, -1:].T, prob_y_batch[:, :-1].T],
        dim=0,
    )

    weighted = logloss * prob_y_reordered
    loss = weighted.sum(dim=0) * pdf

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
