import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler









def plot_esd_for_feature(H_train, H_test):

    H_train_reduced, H_test_reduced = _remove_main_component(H_train, H_test, 250)
    H_svd,_ = _apply_pca(H_train, H_test, 250)
    #scaler = StandardScaler()
    #H_train_reduced = scaler.fit_transform(H_train_reduced)
   # H_test_reduced = scaler.transform(H_test_reduced)

    #scaler = StandardScaler()
    #H_svd = scaler.fit_transform(H_svd)


    print('reduced shape', H_train_reduced.shape)
    print('mean each row', np.mean(H_train_reduced, axis=1))
    print('std each row', np.std(H_train_reduced, axis=1))

    d = H_train.shape[1]
    m = H_train.shape[0]
    H_reduced = 1/m * H_train_reduced.T @ H_train_reduced
    H_svd = 1/m * H_svd.T @ H_svd

    eigenvalues_reduced = np.linalg.eigvals(H_reduced)
    eigenvalues_svd = np.linalg.eigvals(H_svd)
    print('reduced eigenvalues', (eigenvalues_reduced))
    print('svd eigenvalues', (eigenvalues_svd))



    
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues_reduced.real, bins=100, density=True, alpha=0.7, color='blue')
    
    # Add labels and title
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title(r'ESD of $\frac{1}{m} \hat{H_{train}}^T \hat{H_{train}}; H_{train} \in \mathbb{R}^{m \times 500} drived from tnah$')
    plt.grid(True, alpha=0.3)
    
    # Create directory if it doesn't exist
    save_dir = os.path.join("mnist_test", "log", "ESD")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    save_path = os.path.join(save_dir, f"esd_histogram.pdf")
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

















