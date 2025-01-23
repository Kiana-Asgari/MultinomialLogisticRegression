import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline




def learn_mle_on_data(X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot, n_features,\
                          data_name='fashion_mnist', n_trials=1, alpha=None, feature_name='RFF'):
    
    if feature_name == 'RFF':
        H_train, H_test = random_fourier_features(X_train, X_test, n_features)
    elif feature_name == 'RF':
        H_train, H_test = random_feature(X_train, X_test, n_features)
    elif feature_name == 'ReLU':  # Add this condition
        H_train, H_test = random_relu_features(X_train, X_test, n_features)
    elif feature_name == 'tanh':
        H_train, H_test = random_tanh_features(X_train, X_test, n_features)
    elif feature_name == 'box_cox':
        H_train, H_test = box_cox_features(X_train, X_test)
    else:
        H_train, H_test = X_train, X_test

    
    logreg = LogisticRegression(
        penalty=None,
        fit_intercept=False,
        solver='lbfgs',       # can also use 'sag' or 'saga' if data is large
        max_iter=500,
    )

    logreg.fit(H_train, y_train)
    coefs_original     = logreg.coef_.copy()        # shape (3, d)
    
    # Get test accuracy (1 - error rate)
    test_error = log_loss(y_test, logreg.predict_proba(H_test))
    
    base_coef = coefs_original[0]
    Theta_hat_skit = coefs_original - base_coef
    R_00_skit = Theta_hat_skit @ Theta_hat_skit.T
    print(f'      for feature {feature_name},test error: {test_error:.4f}\n   with sklearn: R_00: {R_00_skit} ')
    return R_00_skit, np.array(Theta_hat_skit), H_train, H_test




def random_feature(X_train, X_test, n_features):
    """
    Apply Random Features transformation to data.
    """
    np.random.seed(42)  
    # Scale weights by 1/sqrt(d_in) where d_in is input dimension
    scale = 1.0 / np.sqrt(X_train.shape[1])
    W = np.random.normal(loc=0.0, scale=scale, size=(X_train.shape[1], n_features))

    H_train = 1/np.sqrt(n_features) * X_train @ W
    H_test = 1/np.sqrt(n_features) * X_test @ W

   # scaler_features = StandardScaler()
   # H_train = scaler_features.fit_transform(H_train)
  #  H_test = scaler_features.transform(H_test)

    return H_train, H_test





def random_fourier_features(X_train, X_test, n_features, gamma=0.1):
    """
    Apply Random Fourier Features transformation to data.
    Parameters:
    - X: Input data of shape (n_samples, n_features)
    - n_features: Number of random features to generate
    - gamma: RBF kernel parameter (controls the spread of the random projections)

    Returns:
    - Transformed data with shape (n_samples, n_features)
    """
    np.random.seed(42)
    # Random projection matrix W
    W = np.random.normal(0,  1/np.sqrt(X_train.shape[1]), size=(X_train.shape[1], n_features))
    # Random bias term b
    b = np.random.uniform(0, 2*np.pi, size=n_features)

    # Apply RFF transformation
    Z_train = np.cos(X_train @ W + b)
    Z_test =  np.cos(X_test @ W + b)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)

    return Z_train, Z_test




def random_relu_features(X_train, X_test, n_features):
    """
    Apply Random ReLU Features transformation to data.
    Parameters:
    - X_train: Training data of shape (n_samples, n_features)
    - X_test: Test data of shape (n_samples, n_features)
    - n_features: Number of random features to generate
    - scale: Scaling factor for the random weights (default=1.0)

    Returns:
    - Z_train, Z_test: Transformed data with shape (n_samples, n_features)
    """
    np.random.seed(42)
    # Random projection matrix W
    W = np.random.normal(0, 1/np.sqrt(n_features), size=(X_train.shape[1], n_features))

    # Apply ReLU transformation
    Z_train = np.maximum(0, X_train @ W)
    Z_test = np.maximum(0, X_test @ W)
    
    # Normalize by sqrt(n_features) to maintain variance
    # Standardize features
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)

    return Z_train, Z_test




def random_tanh_features(X_train, X_test, n_features, scale=1):
    """
    Apply Random Tanh Features transformation to data.
    Parameters:
    - X_train: Training data of shape (n_samples, n_features)
    - X_test: Test data of shape (n_samples, n_features)
    - n_features: Number of random features to generate
    - scale: Scaling factor for the random weights (default=1.0)

    Returns:
    - Z_train, Z_test: Transformed data with shape (n_samples, n_features)
    """
    np.random.seed(42)
    
    # Scale weights by 1/sqrt(d_in) where d_in is input dimension
    input_scale = 1.0 / np.sqrt(X_train.shape[1])
    
    # Random projection matrix W
    W = np.random.normal(0, input_scale, size=(X_train.shape[1], n_features))
    # Random bias term b
    #b = np.random.normal(0, input_scale, size=n_features)
    b = np.zeros(n_features)
    # Apply tanh transformation
    Z_train = np.tanh(scale * (X_train @ W + b))
    Z_test = np.tanh(scale * (X_test @ W + b))
    
    # Standardize features
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)

    return Z_train, Z_test




def box_cox_features(X_train, X_test, epsilon=1e-8):
    """
    Apply Box-Cox transformation using sklearn's PowerTransformer.
    
    Parameters:
    - X_train: Training data
    - X_test: Test data
    - epsilon: Small constant to add to make data positive
    
    Returns:
    - X_train_transformed, X_test_transformed: Transformed data
    """
    from sklearn.preprocessing import PowerTransformer
    
    # Make data positive
    X_train_pos = X_train - np.min(X_train, axis=0) + epsilon
    X_test_pos = X_test - np.min(X_test, axis=0) + epsilon
    
    # Apply Box-Cox
    pt = PowerTransformer(method='box-cox')
    X_train_transformed = pt.fit_transform(X_train_pos)
    X_test_transformed = pt.transform(X_test_pos)

    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train_transformed)
    X_test_transformed = scaler.transform(X_test_transformed)
    
    return X_train_transformed, X_test_transformed
