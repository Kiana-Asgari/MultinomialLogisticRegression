import numpy as np
import matplotlib.pyplot as plt
import os



def check_labels(y_train, y_test, X_train, X_test, classes_to_keep=[2,4,6]):
    """
    Check if the labels are balanced and only contain values from classes_to_keep.
    """
    #check the number of samples match the number of features
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Number of samples in X and Y do not match")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Number of features in X_train and X_test do not match")
    print(f'we have {X_train.shape[0]} trainging samples and {X_test.shape[0]} test samples of dimension {X_train.shape[1]}')
    
    # Check train labels
    train_unique = np.unique(y_train)
    if not all(label in classes_to_keep for label in train_unique):
        raise ValueError(f"Training labels contain values outside classes_to_keep {classes_to_keep}")
    
    # Check test labels 
    test_unique = np.unique(y_test)
    if not all(label in classes_to_keep for label in test_unique):
        raise ValueError(f"Test labels contain values outside classes_to_keep {classes_to_keep}")
        
    # Print label distributions
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print(f"Train label counts: {dict(zip(train_unique, train_counts))}")
    print(f"Test label counts: {dict(zip(test_unique, test_counts))}")




def check_features_are_standardized(X_train):
    """
    Check if the features are standardized.
    """
    col_mean = np.mean(X_train, axis=0)
    col_std = np.std(X_train, axis=0)

    if np.max(np.abs(col_mean)) > 1e-5:
        print("WARNING: Features are not standardized(mean)")

    if np.abs(np.max(col_std) - 1) > 1e-3:
        print("WARNING: Features are not standardized(std)")





def plot_esd_for_feature(H_train, alpha = 4):
    np.random.seed(51)  # For reproducibility
    d=H_train.shape[1]
    m=d*alpha
    indices = np.random.choice(H_train.shape[0], size=m, replace=False)
    H_train_reduced = H_train[indices]


    H_reduced = 1/m * H_train_reduced.T @ H_train_reduced
    eigenvalues_reduced = np.linalg.eigvals(H_reduced)    
    
    # Create histogram and plot MP distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues_reduced.real, bins=100, density=True, alpha=0.7,
             color='blue', label=f'Empirical for $\\alpha={alpha}$')

    # Plot Marchenko-Pastur PDF
    for q in [1/alpha]:
        x = np.linspace(0, 4, 1000)
        mp_pdf = marchenko_pastur_pdf(x, q=q)
        plt.plot(x, mp_pdf, 'r-', lw=2, label=f'Marchenko-Pastur, $\\alpha={1/q:.2f}$')
    
    # Add labels and title
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title(f'ESD of $\\frac{{1}}{{m}} H^T H$ from subsample of {m}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create directory if it doesn't exist
    save_dir = os.path.join("real_data", "feature_selection", "log", "ESD")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(save_dir, f"esd_histogram_feature.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"ESD plot saved to {save_path}")
    print(f"Matrix shape: {H_train.shape}")
    print(f"Eigenvalue range: [{eigenvalues_reduced.real.min():.2f}, {eigenvalues_reduced.real.max():.2f}]")





def marchenko_pastur_pdf(x,q, sigma=1):
    """
    Compute the Marchenko-Pastur PDF.
    
    Parameters:
    - x: Points at which to evaluate the PDF
    - q: Ratio of dimensions (p/n, where p is the number of variables, n is the number of samples)
    - sigma: Standard deviation of the entries in the covariance matrix
    
    Returns:
    - Marchenko-Pastur PDF values for the given x
    """


    b = sigma**2 * (1 + np.sqrt(q))**2
    a = sigma**2 * (1 - np.sqrt(q))**2
    
    pdf = np.zeros_like(x)
    mask = (x >= a) & (x <= b)
    pdf[mask] = (1 / (2 * np.pi * sigma**2 * x[mask] * q)) * \
                np.sqrt((b - x[mask]) * (x[mask] - a))
    
    return pdf