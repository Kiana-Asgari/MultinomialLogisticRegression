import math
import numpy as np
import torch

from multinomial_logistic.utils import batched_mlogit


def generate_data(alpha, d, k, Theta_0, random_state=0):
    if random_state == 0:
        random_state = np.random.randint(0, 1000000)
    np.random.seed(random_state)
    n = np.ceil(alpha * d).astype(int)
    X = np.random.randn(n, d)  # shape (n, d)

    # Logits for non-baseline classes: shape (n, k)
    Beta_batch = X @ Theta_0.T
    prob_y_batch = batched_mlogit(Beta_batch)
    cdf_y_batch = np.cumsum(prob_y_batch, axis=1)

    # Draw labels
    Y_onehot = np.zeros((n, k))
    random_values = np.random.uniform(0, 1, size=(n,))
    samples = np.argmax(random_values[:, None] <= cdf_y_batch, axis=1)

    baseline_mask = samples < k
    Y_onehot[np.arange(n)[baseline_mask], samples[baseline_mask]] = 1.0

    return X, Y_onehot


def generate_data_torch(
    alpha,
    d,
    k,
    Theta_0,
    random_state=0,
    device=None,
    dtype=torch.float32,
):
    if alpha is None:
        raise ValueError("alpha must be provided when using generate_data_torch.")

    if device is None:
        if torch.is_tensor(Theta_0):
            device = Theta_0.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    generator = torch.Generator(device=device.type)
    if random_state == 0:
        random_state = torch.randint(0, 1_000_000, (1,), device=device).item()
    generator.manual_seed(int(random_state))

    n = max(1, int(math.ceil(alpha * d)))
    X = torch.randn((n, d), generator=generator, device=device, dtype=dtype)
    theta = Theta_0 if torch.is_tensor(Theta_0) else torch.as_tensor(Theta_0, dtype=dtype, device=device)

    Beta_batch = X @ theta.T
    padded = torch.cat(
        (Beta_batch, torch.zeros((n, 1), dtype=dtype, device=device)),
        dim=1,
    )
    prob_y_batch = torch.softmax(padded, dim=1)
    samples = torch.multinomial(prob_y_batch, num_samples=1, replacement=True, generator=generator).squeeze(1)

    Y_onehot = torch.zeros((n, k), dtype=dtype, device=device)
    baseline_mask = samples < k
    if baseline_mask.any():
        indices = samples[baseline_mask]
        Y_onehot[baseline_mask, indices] = 1.0

    return X, Y_onehot
