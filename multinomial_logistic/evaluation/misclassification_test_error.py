import torch

from state_evolution.utils import sphere_mesh_integration, get_primary_device








def misclassification_test_error(S, R_00, schur_t, A_t, alpha, k, k_0, seed=42):
    breakpoint()
    return 0
    
    device = get_primary_device()
    dtype = torch.float32

    R_00_tensor = torch.tensor(R_00, device=device, dtype=dtype)
    schur_tensor = torch.tensor(schur_t, device=device, dtype=dtype)
    A_tensor = torch.tensor(A_t, device=device, dtype=dtype)
    R00_sqrt = _matrix_sqrt(R_00_tensor)
    schur_root = _matrix_sqrt(schur_tensor)

    # Pre-transpose matrices once instead of in every integrand call
    R00_sqrt_t = R00_sqrt.transpose(-1, -2)
    schur_root_t = schur_root.transpose(-1, -2)
    A_t_transposed = A_tensor.transpose(-1, -2)

    accuracy_tensor = sphere_mesh_integration(
        _misclassification_integrand,
        R00_sqrt_t,
        schur_root_t,
        A_t_transposed,
        k,
        k_0,
        input_dim=k + k_0,
        output_dim=1,
        seed=seed,
        n_radius=18,
        n_polar=14,
        radius=5,
        batch_size=350_000,
    )

    accuracy = accuracy_tensor[0]
    print("     accuracy_mesh: ", accuracy.item())
    return (1 - accuracy).item()






def _misclassification_integrand(Z_batch, R00_sqrt_t, schur_root_t, A_t, k, k_0):
    g_batch, g_0_batch = _coloring_transform(Z_batch, A_t, R00_sqrt_t, schur_root_t, k, k_0)
    pdf = _standard_normal_pdf(Z_batch)
    prob_y0_batch = _batched_mlogit(g_0_batch)

    # Optimized argmax: avoid creating zero column and concatenation
    # Concatenating [g_batch, zeros] means: if max(g_batch) > 0, argmax is from g_batch;
    # if max(g_batch) <= 0, argmax is the first zero in g_batch or k (the zeros column)
    # For efficiency, if g_max <= 0, we use k (matches behavior when no zeros in g_batch)
    g_max, g_argmax = g_batch.max(dim=1)
    y1_batch = torch.where(g_max > 0, g_argmax, k)

    # Optimized gather: use advanced indexing instead of gather with unsqueeze/squeeze
    # Use torch.arange with explicit dtype=torch.long for indexing
    batch_size = prob_y0_batch.shape[0]
    indices = torch.arange(batch_size, dtype=torch.long, device=prob_y0_batch.device)
    prob_y0_of_y1 = prob_y0_batch[indices, y1_batch]

    integrand = prob_y0_of_y1 * pdf
    return integrand.unsqueeze(1)



############################

@torch.no_grad()
def _coloring_transform(Z_batch, A_t, R00_sqrt_t, schur_root_t, k, k_0):
    Z_top = Z_batch[:, :k]
    Z_bottom = Z_batch[:, -k_0:]
    g = torch.matmul(Z_bottom, A_t) + torch.matmul(Z_top, schur_root_t)
    g_0 = torch.matmul(Z_bottom, R00_sqrt_t)
    return g, g_0

@torch.no_grad()
def _matrix_sqrt(matrix):
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
    return eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)

@torch.no_grad()
def _standard_normal_pdf(samples):
    d = samples.shape[-1]
    norm_sq = (samples ** 2).sum(dim=-1)
    coeff = (2 * torch.pi) ** (-0.5 * d)
    return coeff * torch.exp(-0.5 * norm_sq)

@torch.no_grad()
def _batched_mlogit(beta):
    # Efficiently create zero column using new_zeros (faster than torch.zeros with explicit device/dtype)
    zeros = beta.new_zeros((beta.shape[0], 1))
    logits = torch.cat([beta, zeros], dim=1)
    return torch.softmax(logits, dim=1)