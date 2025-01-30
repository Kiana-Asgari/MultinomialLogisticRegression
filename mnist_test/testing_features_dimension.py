import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mnist_test.train_and_eval import esd_empirical
from mnist_test.log.log_fp_mnist import evaluate_mle



def plot_esd_for_hessian(alpha,
                        n_hidden,\
                        R_00, Theta_0, k, k_0, \
                        H_train, y_train, H_test, y_test, \
                        n_iter, tol, y_train_full, y_test_full,):
    print(f"******* plot_esd_for_hessian, alpha: {alpha}*******")
    np.random.seed(51)  # For reproducibility


    test_errors, train_errors, misclass_test_errors, f_norms, eigvals = evaluate_mle(alpha=alpha,
                                    n_hidden=n_hidden,\
                                    R_00=R_00, Theta_0=Theta_0, k=2, k_0=2, \
                                    X_train=H_train, y_train=y_train, X_test=H_test, y_test=y_test, \
                                    n_iter=1, tol=1e-4, y_train_full=y_train, y_test_full=y_test, plot_esd=True)
    print('test_errors: ', test_errors)
    print('train_errors: ', train_errors)
    print('misclass_test_errors: ', misclass_test_errors)
    print('f_norms: ', f_norms)
    print(f"******* eigvals shape: {eigvals.shape}*******")
    print(f"******* eigvals min: {eigvals.min():.2f}, max: {eigvals.max():.2f}*******")
     # Create histogram and plot MP distribution
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of eigenvalues
    plt.hist(eigvals, bins=100, density=True, alpha=0.7,
             color='blue', label=f'ESD for $\\alpha={alpha}$')

    
    # Add labels and title
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title(f'ESD')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create directory if it doesn't exist
    save_dir = os.path.join("mnist_test", "log", "ESD", "hessian")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(save_dir, f"esd_histogram.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"ESD plot saved to {save_path}")
    print(f"Matrix shape: {H_train.shape}")
    print(f"Eigenvalue range: [{eigvals.min():.2f}, {eigvals.max():.2f}]")






def plot_esd_for_feature(H_train, H_test, alpha = 4):
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
    save_dir = os.path.join("mnist_test", "log", "ESD")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(save_dir, f"esd_histogram_feature.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"ESD plot saved to {save_path}")
    print(f"Matrix shape: {H_train.shape}")
    print(f"Eigenvalue range: [{eigenvalues_reduced.real.min():.2f}, {eigenvalues_reduced.real.max():.2f}]")





def _apply_pca(H_train, H_test, n_components=100):
    # Step 1: Re-center and normalize X_train
   # scaler = StandardScaler()
   # H_train_normalized = scaler.fit_transform(H_train)
    
    # Step 2: Apply PCA to reduce dimensionality of X_train
    pca = PCA(n_components=n_components)
    H_train_reduced = pca.fit_transform(H_train)
    
    # Step 3: Transform X_test using the same PCA and scaler
    #H_test_normalized = scaler.transform(H_test)
    H_test_reduced = pca.transform(H_test)
    
    return H_train_reduced, H_test_reduced



def _remove_main_component(H_train, H_test, n_lower_components=10):
    """
    Reduces the dimensionality of the feature matrix by projecting onto the top k singular eigenvectors.
    
    Parameters:
        feature_matrix (numpy.ndarray): The input feature matrix of shape (n_samples, n_features).
        k (int): The number of top singular vectors to project onto.
        
    Returns:
        numpy.ndarray: The reduced-dimensionality feature matrix of shape (n_samples, k).
    """
    # Perform Singular Value Decomposition
    #U, S, Vt = np.linalg.svd(H_train, full_matrices=False)
    pca = PCA()
    pca.fit(H_train)

    #singular_values = pca.singular_values_
    all_components = pca.components_
    lower_components = all_components[-n_lower_components:]
    

    # Project the feature matrix onto the bottom k singular vectors
    H_train_reduced = np.dot(H_train, lower_components.T)
    H_test_reduced = np.dot(H_test, lower_components.T)
    
    return H_train_reduced, H_test_reduced



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

















