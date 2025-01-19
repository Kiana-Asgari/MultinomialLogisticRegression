import numpy as np
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import batched_mlogit

def generate_data(alpha, d, k, Theta_0, random_state=0):
    if random_state==0:
        random_state = np.random.randint(0, 1000000) 
    np.random.seed(random_state)
    n = np.ceil(alpha*d).astype(int)
    X = np.random.randn(n, d)  # shape (n, d)
    
    # Logits for non-baseline classes: shape (n, k)
    Beta_batch = X @ Theta_0.T
    prob_y_batch = batched_mlogit(Beta_batch)
    cdf_y_batch = np.cumsum(prob_y_batch, axis=1)

    # Prob(class j) for j=1..k    
    # Draw labels
    Y_onehot = np.zeros((n, k))  # shape (n, k)
    random_values = np.random.uniform(0, 1, size=(n,))
    samples = np.argmax(random_values[:, None] <= cdf_y_batch, axis=1) # e_j => label j-1, 0 => label k

    # convert to one-hot
    Y_onehot = np.zeros((n, k))  # shape (n, k)
    # Only set 1.0 for non-baseline classes (when samples < k)
    baseline_mask = samples < k
    Y_onehot[np.arange(n)[baseline_mask], samples[baseline_mask]] = 1.0

    return X, Y_onehot
