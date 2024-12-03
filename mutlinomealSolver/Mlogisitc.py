import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Parameters
alphas = 4
num_classes = 4
n_samples = 4000
lambda_values = np.arange(0.5, 100, 0.1)  # Different sample sizes to evaluate
test_errors = []

# Iterate over different sample sizes
for lambda_reg in lambda_values:
    d = n_samples // alphas  # Dimension of feature space
    np.random.seed(42)
    X = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n_samples)

    # Generate normalized true underlying parameters (one vectors)
    theta_star = np.ones((num_classes-1, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True)  # Normalize to unit vectors

    # Compute logits for each class
    logits = X @ theta_star.T

    # Apply softmax to get probabilities
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    probabilities = softmax(logits)

    # Assign labels based on probabilities
    y = np.array([np.random.choice(num_classes, p=prob) for prob in probabilities])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the multinomial logistic regression model with L2 regularization
    model = LogisticRegression(
        multi_class='multinomial',
        penalty='l2',
        C=1 / lambda_reg,  # Inverse of regularization strength
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict probabilities on test data
    y_prob = model.predict_proba(X_test)

    # Compute log loss (cross-entropy loss) as the test error
    error = log_loss(y_test, y_prob)
    test_errors.append(error)
    print(f"n={n_samples}, Test Error (Log Loss): {error:.4f}")

# Plot Test Error vs. Training Set Size
plt.figure(figsize=(8, 6))
plt.plot(lambda_values, test_errors, marker='o', linestyle='-')
plt.xlabel('Value of regularizer')
plt.ylabel('Test Error (Log Loss)')
plt.title('Test Error vs. Number of Samples')
plt.grid(True)
plt.show()
