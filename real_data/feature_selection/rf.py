import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh



def feature_selection(X_train, X_test, n_features, method='relu', decorrelate=True, seed=42):
    np.random.seed(seed)

    if method == 'relu':
        print("Using ReLU features...")
        X_train, X_test = random_relu_features(X_train, X_test, n_features)
    elif method == 'tanh':
        print("Using Tanh features...")
        X_train, X_test = random_tanh_features(X_train, X_test, n_features)
    else:
        raise ValueError(f" feature selection method {method} not supported :(")
    
    if decorrelate:
        print("Decorrelating features...")
        X_train, X_test = decorrelated_features(X_train, X_test, n_features)


    return X_train, X_test





def random_relu_features(X_train, X_test, n_features, scale=.1):

    # Normalize by sqrt(n_features) to maintain variance
    W = np.random.normal(0, scale/np.sqrt(X_train.shape[1]), size=(X_train.shape[1], n_features))
    H_train = np.maximum(0, X_train @ W)
    H_test = np.maximum(0, X_test @ W)
    
    # Standardize features
    scaler = StandardScaler(with_mean=True, with_std=True)
    H_train = scaler.fit_transform(H_train)
    H_test = scaler.transform(H_test)

    return H_train, H_test




def random_tanh_features(X_train, X_test, n_features):
    
    # Scale weights by 1/sqrt(d_in) where d_in is input dimension
    input_scale = 1.0 / np.sqrt(X_train.shape[1])
    
    # Random projection matrix W
    W = np.random.normal(0, input_scale, size=(X_train.shape[1], n_features))
    # Random bias term b
    #b = np.random.normal(0, input_scale, size=n_features)

    H_train = np.tanh( X_train @ W )
    H_test = np.tanh( X_test @ W )
    
    # Standardize features
    scaler = StandardScaler()
    H_train = scaler.fit_transform(H_train)
    H_test = scaler.transform(H_test)

    return H_train, H_test



def decorrelated_features(X_train, X_test, n_features):
    # Eigenvalue decomposition (since cov is symmetric)
    cov = np.cov(X_train, rowvar=False)
    eigvals, eigvecs = eigh(cov) 
    eigvals[eigvals < 1e-12] = 1e-12  # Avoid numerical issues

    # Compute inverse square root
    inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals))
    inv_sqrt_cov = eigvecs @ inv_sqrt_eigvals @ eigvecs.T  # Reconstruct matrix
    
    H_train = X_train @ inv_sqrt_cov
    H_test = X_test @ inv_sqrt_cov

    return H_train, H_test
