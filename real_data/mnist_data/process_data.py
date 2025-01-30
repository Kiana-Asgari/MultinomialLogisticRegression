import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import fashion_mnist





def get_cleaned_fashion_mnist_data(classes_to_keep):
    """
    Load and preprocess the Fashion MNIST dataset to keep only specified classes.
    
    Args:
        classes_to_keep (list): List of integer class labels to keep (e.g., [0, 1, 2]).
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test) with filtered data.
        features are normalized and labels are inside the classes_to_keep (NOT one-hot encoded)
    """

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test, classes_to_keep)

    return (x_train, y_train), (x_test, y_test)




def preprocess(x_train, y_train, x_test, y_test, classes_to_keep):  
    # Filter the training data
    train_filter = np.isin(y_train, classes_to_keep)
    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]

    
    # Filter the test data
    test_filter = np.isin(y_test, classes_to_keep)
    x_test_filtered = x_test[test_filter]
    y_test_filtered = y_test[test_filter]

    # Normalize remaining pixels

    scaler = StandardScaler()
    x_train_normalized_filtered = scaler.fit_transform(x_train_filtered)
    x_test_normalized_filtered = scaler.transform(x_test_filtered)  

  
    return (x_train_normalized_filtered, y_train_filtered), (x_test_normalized_filtered, y_test_filtered)

