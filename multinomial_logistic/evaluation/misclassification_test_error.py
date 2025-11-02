import torch

from state_evolution.utils import mesh_integration, ensure_tensor








def misclassification_test_error(S, R_00, schur_t, A_t, alpha, k, k_0, seed=42):
    device = torch.device("cuda")

    source_tensors = [
        arg
        for arg in (S, R_00, schur_t, A_t, alpha)
        if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg)
    ]
    dtype = source_tensors[0].dtype if source_tensors else torch.float64

    def to_tensor(arg):
        return ensure_tensor(arg, dtype=dtype, device=device)

    S_tensor = to_tensor(S)
    R_00_tensor = to_tensor(R_00)
    schur_tensor = to_tensor(schur_t)
    A_tensor = to_tensor(A_t)
    alpha_tensor = to_tensor(alpha)

    R00_sqrt = _matrix_sqrt(R_00_tensor)
    schur_root = _matrix_sqrt(schur_tensor)

    accuracy_tensor = mesh_integration(
        _misclassification_integrand,
        R00_sqrt,
        schur_root,
        A_tensor,
        k,
        k_0,
        input_dim=k + k_0,
        output_dim=1,
        seed=seed,
        n_mesh=45,
        size=8,
    )

    accuracy = accuracy_tensor[0]
    print("     accuracy_mesh: ", accuracy.item())
    return (1 - accuracy).item()






def _misclassification_integrand(Z_batch, R00_sqrt, schur_root, A_t, k, k_0):

    g_batch, g_0_batch = _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0)
    pdf = _standard_normal_pdf(Z_batch)
    prob_y0_batch = _batched_mlogit(g_0_batch)

    logits = torch.cat(
        [g_batch, torch.zeros((g_batch.shape[0], 1), device=g_batch.device, dtype=g_batch.dtype)],
        dim=1,
    )
    y1_batch = torch.argmax(logits, dim=1)
    prob_y0_of_y1 = prob_y0_batch.gather(1, y1_batch.unsqueeze(1)).squeeze(1)

    integrand = prob_y0_of_y1 * pdf


    return integrand.unsqueeze(1)



############################

@torch.no_grad()
def _coloring_transform(Z_batch, A_t, R00_sqrt, schur_root, k, k_0):
    Z_top = Z_batch[:, :k]
    Z_bottom = Z_batch[:, -k_0:]
    g = torch.matmul(Z_bottom, A_t.transpose(-1, -2)) + torch.matmul(Z_top, schur_root.transpose(-1, -2))
    g_0 = torch.matmul(Z_bottom, R00_sqrt.transpose(-1, -2))
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
    zeros = torch.zeros((beta.shape[0], 1), dtype=beta.dtype, device=beta.device)
    logits = torch.cat([beta, zeros], dim=1)
    return torch.softmax(logits, dim=1)