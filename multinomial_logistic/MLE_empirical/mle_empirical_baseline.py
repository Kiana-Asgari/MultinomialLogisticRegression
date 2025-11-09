import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from multinomial_logistic.MLE_empirical.utils.data_generation import generate_data, generate_data_torch


_DEVICE = torch.device("cuda:3")
_DTYPE = torch.float32


def _to_tensor(arr):
    return torch.as_tensor(arr, dtype=_DTYPE, device=_DEVICE)


def _prepare_covariance(covariance, size):
    cov_tensor = _to_tensor(covariance)
    if cov_tensor.ndim == 2:
        if cov_tensor.shape[0] != size or cov_tensor.shape[1] != size:
            raise ValueError(f"Expected covariance shape ({size}, {size}), got {tuple(cov_tensor.shape)}.")
    else:
        raise ValueError("Covariance tensor must be scalar, vector, or 2D matrix.")
    return cov_tensor


def _dispatch(original, tensor):
    return {True: tensor, False: tensor.detach().cpu().numpy()}[torch.is_tensor(original)]


def log_sum_exp_batch(V):
    logits = _to_tensor(V)
    augmented = torch.cat((torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device), logits), dim=1)
    values = torch.logsumexp(augmented, dim=1)
    return _dispatch(V, values)


def batched_mlogit(beta):
    logits = _to_tensor(beta)
    padded = torch.cat((logits, torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device)), dim=1)
    return F.softmax(padded, dim=1)


def batched_mlogit_jacobian(beta):
    probs = batched_mlogit(beta)[:, :-1]
    diag = torch.diag_embed(probs)
    outer = torch.einsum("ni,nj->nij", probs, probs)
    return diag - outer


def _matrix_sqrt(matrix):
    mat = _to_tensor(matrix)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (mat + mat.T))
    sqrt_diag = torch.where(eigvals > 0, eigvals.sqrt(), torch.zeros_like(eigvals))
    return eigvecs @ torch.diag_embed(sqrt_diag) @ eigvecs.transpose(-1, -2)


def negative_log_likelihood_and_gradient(theta, X, Y, lambda_reg):
    logits = X @ theta.T
    log_probs = torch.logsumexp(torch.cat((torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device), logits), dim=1), dim=1)
    target_term = torch.einsum("ni,ni->", Y, logits)
    loss = (log_probs.sum() - target_term + 0.5 * lambda_reg * torch.linalg.norm(theta) ** 2) / X.shape[0]
    probs = batched_mlogit(logits)[:, :-1]
    grad_part = torch.einsum("ni,nj->ij", probs, X) - torch.einsum("ni,nj->ij", Y, X)
    gradient = grad_part / X.shape[0] + lambda_reg * theta
    return loss, gradient


def lbfgs_multinomial(X, Y, lambda_reg, theta_init=None, max_iter=5000, tol=1e-5, verbose=False):
    X_tensor = _to_tensor(X)
    Y_tensor = _to_tensor(Y)
    k = Y_tensor.shape[1]
    d = X_tensor.shape[1]
    theta0 = _to_tensor(theta_init) if theta_init is not None else torch.randn((k, d), dtype=_DTYPE, device=_DEVICE)
    theta_norm = torch.linalg.norm(theta0)
    theta_data = theta0 / torch.clamp(theta_norm, min=1e-12)
    theta_param = torch.nn.Parameter(theta_data.clone())
    optimizer = torch.optim.LBFGS([theta_param], max_iter=max_iter, tolerance_grad=tol, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss, grad = negative_log_likelihood_and_gradient(theta_param, X_tensor, Y_tensor, lambda_reg)
        theta_param.grad = grad.detach()
        history.append(loss.detach().cpu())
        return loss

    optimizer.step(closure)
    state = optimizer.state[theta_param]
    result = SimpleNamespace(
        nit=state.get("n_iter", 0),
        nfev=state.get("func_evals", 0),
        success=True,
        status=0,
        message="torch.optim.LBFGS"
    )
    return theta_param.detach(), result, history


def _logistic_loss(theta, X, Y):
    logits = X @ theta.T
    loss = torch.logsumexp(torch.cat((torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device), logits), dim=1), dim=1)
    target = torch.einsum("ni,ni->", Y, logits)
    return (loss.sum() - target) / X.shape[0]


def _classification_error(theta, X, Y):
    logits = X @ theta.T
    scores = torch.cat((logits, torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device)), dim=1)
    preds = torch.argmax(scores, dim=1)
    baseline_indicator = (Y.sum(dim=1, keepdim=True) == 0).to(Y.dtype)
    labels = torch.argmax(torch.cat((Y, baseline_indicator), dim=1), dim=1)
    return 1.0 - (preds == labels).to(torch.float32).mean()


def fit_mle_baseline(alpha=None, k=None, lambda_reg=0, R_00=None, n_trials=1, d=500, return_full_results=True, X_train_batch=None, Y_train_batch=None, X_test_batch=None, Y_test_batch=None):
    learn_from_data = X_train_batch is not None
    total_trials = 1 if learn_from_data else n_trials
    dim = X_train_batch.shape[1] if learn_from_data else d
    classes = Y_train_batch.shape[1] if learn_from_data else k
    base_cov = _prepare_covariance(R_00, classes)
    sqrt_block = _matrix_sqrt(base_cov)
    zeros_pad = torch.zeros((classes, dim - classes), dtype=_DTYPE, device=_DEVICE)
    Theta_0 = torch.cat((sqrt_block, zeros_pad), dim=1) if not learn_from_data else torch.zeros((classes, dim), dtype=_DTYPE, device=_DEVICE)
    theta_collection = torch.zeros((total_trials, classes, dim), dtype=_DTYPE, device=_DEVICE)
    norms = torch.zeros((total_trials,), dtype=_DTYPE, device=_DEVICE)
    test_errors = torch.zeros((total_trials,), dtype=_DTYPE, device=_DEVICE)
    train_errors = torch.zeros((total_trials,), dtype=_DTYPE, device=_DEVICE)
    misclassification = torch.zeros((total_trials,), dtype=_DTYPE, device=_DEVICE)

    for idx in range(total_trials):
        batch_seed = 2 * idx
        if learn_from_data:
            X_tensor = _to_tensor(X_train_batch)
            Y_tensor = _to_tensor(Y_train_batch)
        else:
            X_tensor, Y_tensor = generate_data_torch(
                alpha=alpha,
                d=dim,
                k=classes,
                Theta_0=Theta_0,
                random_state=batch_seed,
                device=_DEVICE,
                dtype=_DTYPE,
            )

        theta_hat, _, history = lbfgs_multinomial(X_tensor, Y_tensor, lambda_reg, verbose=False)
        theta_collection[idx] = theta_hat
        norms[idx] = torch.linalg.norm(Theta_0 - theta_hat) ** 2
        train_errors[idx] = _logistic_loss(theta_hat, X_tensor, Y_tensor)

        if X_test_batch is None:
            eval_X, eval_Y = generate_data_torch(
                alpha=alpha,
                d=dim,
                k=classes,
                Theta_0=Theta_0,
                random_state=batch_seed + 1,
                device=_DEVICE,
                dtype=_DTYPE,
            )
        else:
            eval_X = _to_tensor(X_test_batch)
            eval_Y = _to_tensor(Y_test_batch)

        test_errors[idx] = _logistic_loss(theta_hat, eval_X, eval_Y)
        misclassification[idx] = _classification_error(theta_hat, eval_X, eval_Y)

    Theta_hats_cpu = theta_collection.detach().cpu().numpy()
    norms_cpu = norms.detach().cpu().numpy()
    test_cpu = test_errors.detach().cpu().numpy()
    train_cpu = train_errors.detach().cpu().numpy()
    mis_cpu = misclassification.detach().cpu().numpy()

    if return_full_results:
        return Theta_hats_cpu, norms_cpu, test_cpu, train_cpu, mis_cpu
    return (
        Theta_hats_cpu.mean(axis=0),
        norms_cpu.mean(),
        test_cpu.mean(),
        train_cpu.mean(),
        mis_cpu.mean()
    )


def esd_empirical(alpha, k, lambda_reg, R_00, max_iter=100, d=250, plot_esd=False):
    base_cov = _prepare_covariance(R_00, k)
    sqrt_block = _matrix_sqrt(base_cov)
    zeros_pad = torch.zeros((k, d - k), dtype=_DTYPE, device=_DEVICE)
    Theta_0 = torch.cat((sqrt_block, zeros_pad), dim=1)
    theta_sum = torch.zeros_like(Theta_0)
    esd_records = []
    print(" starting fitting mle with lambda_reg:", lambda_reg, "for alpha:", alpha, "and k:", k)

    for iteration in range(max_iter):
        X_tensor, Y_tensor = generate_data_torch(
            alpha=alpha,
            d=d,
            k=k,
            Theta_0=Theta_0,
            random_state=iteration,
            device=_DEVICE,
            dtype=_DTYPE,
        )
        theta_hat, _, _ = lbfgs_multinomial(X_tensor, Y_tensor, lambda_reg, verbose=False)
        theta_sum += theta_hat
        hessian = batched_hessian(theta_hat, X_tensor, Y_tensor)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        esd_records.append(eigenvalues.detach().cpu())
        sample_norm = torch.linalg.norm(X_tensor, dim=1).mean().item()
        print("norm of X:", sample_norm)
        print("Hessian shape:", tuple(hessian.shape))
        print("first diagonal:", hessian.diagonal().detach().cpu().numpy())
        print("first row:", hessian[0].detach().cpu().numpy())
        print("smallest and largest eigenvalues:", float(eigenvalues.min()), float(eigenvalues.max()))
        print("determinant:", float(torch.det(hessian)))
        hist500 = torch.histc(eigenvalues, bins=500, min=float(eigenvalues.min()), max=float(eigenvalues.max()))
        density500 = hist500.max().item()
        print(f"Iteration {iteration + 1} completed, bins=500 max density proxy: {density500:.4f}")
        hist100 = torch.histc(eigenvalues, bins=100, min=float(eigenvalues.min()), max=float(eigenvalues.max()))
        density100 = hist100.max().item()
        print(f"Iteration {iteration + 1} completed, bins=100 max density proxy: {density100:.4f}")

    esd_tensor = torch.stack(esd_records)
    avg_esd = esd_tensor.mean(dim=0).cpu().numpy()
    avg_theta_hat = (theta_sum / max_iter).detach().cpu().numpy()

    if plot_esd:
        plt.figure(figsize=(10, 6))
        plt.hist(avg_esd, bins=100, density=True, facecolor='none', edgecolor='red')
        plt.xlabel('Eigenvalues')
        plt.ylabel('Probability Density')
        plt.title(f'Eigenvalue Spectrum Density (α={alpha}, k={k}, R_00={R_00})')
        save_path = f'multinomial_logistic/data/ESD/{k}_classes/esd_alpha_{alpha}_k_{k}_R_00_{R_00}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    return avg_theta_hat, avg_esd, esd_tensor.cpu().numpy()


def batched_hessian(theta, X, Y):
    logits = X @ theta.T
    jacobian = batched_mlogit_jacobian(logits)
    scatter = torch.einsum("ni,nj->nij", X, X)
    n = X.shape[0]
    jacobian_flat = jacobian.reshape(n, -1)
    scatter_flat = scatter.reshape(n, -1)
    kron_sum = jacobian_flat.transpose(0, 1) @ scatter_flat
    k = theta.shape[0]
    d = X.shape[1]
    hessian = kron_sum.reshape(k, k, d, d).permute(0, 2, 1, 3).reshape(k * d, k * d) / n
    return hessian
