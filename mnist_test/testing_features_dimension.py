import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler









def plot_esd_for_feature(H_train, H_test):

    H_reduced, effective_dim = _apply_pca(H_train, H_test, 50)
    print('reduced shape', H_reduced.shape)
    print('effective dim', effective_dim)
    print('H_train shape', H_train.shape)
    print('H_test shape', H_test.shape)
    print('H_train mean', np.mean(H_train, axis=0))
    print('H_train std', np.std(H_train, axis=0))
    print('H_test mean', np.mean(H_test, axis=0))
    print('H_test std', np.std(H_test, axis=0))
    # Calculate H^T H
    d = H_train.shape[1]
    m = H_train.shape[0]
    H = 1/m * H_reduced.T @ H_reduced
    print('H shape', H.shape)
    print('d, m', d, m)
    
    # Get eigenvalues and filter for those less than 5
    eigenvalues = np.linalg.eigvals(H)
    print('len eigenvalues', len(eigenvalues))
    eigenvalues = eigenvalues[(eigenvalues < 8) & (eigenvalues > 1e-2)]
    print('len eigenvalues', len(eigenvalues))


    
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues.real, bins=100, density=True, alpha=0.7, color='blue')
    
    # Add labels and title
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title('Empirical Spectrum Distribution of H^T H')
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
    print(f"Eigenvalue range: [{eigenvalues.real.min():.2f}, {eigenvalues.real.max():.2f}]")





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
















