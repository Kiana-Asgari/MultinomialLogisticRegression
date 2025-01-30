from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_cleaned_mnist_data(classes_to_keep=[0,1,2], std_threshold=1e-4, normalize=True, pca=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return preprocess_mnist(x_train, y_train, x_test, y_test, classes_to_keep, std_threshold, normalize, pca)


def get_cleaned_fashion_mnist_data(classes_to_keep=[0,1,2], 
                                   std_threshold=1e-4,
                                     normalize=True, 
                                     pca=False,
                                     low_pass_filter=False):

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if low_pass_filter:
        cutoff_frequency = 10
        x_train = apply_low_pass_to_dataset(x_train, cutoff=cutoff_frequency)
        x_test = apply_low_pass_to_dataset(x_test, cutoff=cutoff_frequency)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # Use the same preprocessing function as regular MNIST
    return preprocess_mnist(x_train, y_train, x_test, y_test, classes_to_keep, std_threshold, normalize, pca)










def preprocess_mnist(x_train, y_train, x_test, y_test, classes_to_keep=[0,1,2], std_threshold=1e-4, normalize=True, pca=False):
    """
    Load and preprocess the MNIST dataset to keep only specified classes.
    
    Args:
        classes_to_keep (list): List of integer class labels to keep (e.g., [0, 1, 2]).
        x_train (np.array): normalized Training data.
        y_train (np.array): normalized Training labels.
        x_test (np.array): normalized Test data.
        y_test (np.array): normalized Test labels.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test) with filtered data.
    """
    
    # Filter the training data
    train_filter = np.isin(y_train, classes_to_keep)
    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]

    
    # Filter the test data
    test_filter = np.isin(y_test, classes_to_keep)
    x_test_filtered = x_test[test_filter]
    y_test_filtered = y_test[test_filter]


    # Convert labels to one-hot encoding
    y_train_filtered_one_hot, y_test_filtered_one_hot = one_hot_encoding(y_train_filtered, y_test_filtered, classes_to_keep)

    # Normalize remaining pixels
    if normalize:
        scaler = StandardScaler()

        # Fit the scaler on the training set and transform it
        x_train_normalized_filtered = scaler.fit_transform(x_train_filtered)

        # Transform the test set using the same mean and std from the training set
        x_test_normalized_filtered = scaler.transform(x_test_filtered)  
    else:
        x_train_normalized_filtered = x_train_filtered
        x_test_normalized_filtered = x_test_filtered
    
    if pca:
        pca_components = PCA(n_components=0.95)
        x_train_pca_filtered = pca_components.fit_transform(x_train_normalized_filtered)
        x_test_pca_filtered = pca_components.transform(x_test_normalized_filtered)

        scaler = StandardScaler()
        # Fit the scaler on the training set and transform it
        x_train_normalized_filtered = scaler.fit_transform(x_train_pca_filtered)

        # Transform the test set using the same mean and std from the training set
        x_test_normalized_filtered = x_test_pca_filtered ## remove this later
        x_train_normalized_filtered = x_train_pca_filtered

        x_test_normalized_filtered = x_test_normalized_filtered[:, :x_train_normalized_filtered.shape[1]]
        print('after pca, shape of x_train_normalized_filtered: ', x_train_normalized_filtered.shape)
  
    return (x_train_normalized_filtered, y_train_filtered, y_train_filtered_one_hot), \
        (x_test_normalized_filtered, y_test_filtered, y_test_filtered_one_hot)



def one_hot_encoding(y_train_filtered, y_test_filtered, classes,):
    """
    Convert labels to one-hot encoding where:
    - If y == classes_to_keep[0], return zero vector
    - If y == classes_to_keep[i], return e_(i-1) for i > 0
    
    Args:
        y_train_filtered: training labels
        y_test_filtered: test labels
        classes: list of classes to keep
    
    Returns:
        Tuple of (train_one_hot, test_one_hot) arrays
    """
    # Get sizes
    n_train = len(y_train_filtered)
    n_test = len(y_test_filtered)
    k = len(classes) - 1  # dimension is one less than number of classes
    
    # Initialize arrays with zeros
    y_train_one_hot = np.zeros((n_train, k))
    y_test_one_hot = np.zeros((n_test, k))
    
    # For training data
    for i in range(1, len(classes)):  # start from 1 since class 0 stays zero
        train_mask = (y_train_filtered == classes[i])
        y_train_one_hot[train_mask, i-1] = 1
        
    # For test data
    for i in range(1, len(classes)):  # start from 1 since class 0 stays zero
        test_mask = (y_test_filtered == classes[i])
        y_test_one_hot[test_mask, i-1] = 1

    return y_train_one_hot, y_test_one_hot



import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from tqdm import tqdm  # For progress visualization

def low_pass_filter_fft(image, cutoff=10):
    """
    Apply a low-pass filter in the frequency domain to a single image.

    Parameters:
    - image: Input 2D image (grayscale).
    - cutoff: Cutoff frequency for the low-pass filter.

    Returns:
    - filtered_image: Filtered image in the spatial domain.
    """
    # Perform 2D FFT and shift zero frequency to the center
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)

    # Create a circular low-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff:
                mask[i, j] = 1

    # Apply the mask
    fft_filtered = fft_shifted * mask

    # Inverse FFT to return to spatial domain
    fft_inverse_shifted = ifftshift(fft_filtered)
    filtered_image = np.abs(ifft2(fft_inverse_shifted))
    
    return filtered_image

def apply_low_pass_to_dataset(dataset, cutoff=10):
    """
    Apply low-pass FFT filtering to all images in a dataset.

    Parameters:
    - dataset: Dataset of images (e.g., X_train or X_test).
    - cutoff: Cutoff frequency for the low-pass filter.

    Returns:
    - filtered_dataset: Filtered dataset.
    """
    filtered_dataset = np.zeros_like(dataset, dtype=np.float32)
    for i in tqdm(range(len(dataset)), desc="Processing Images"):
        filtered_dataset[i] = low_pass_filter_fft(dataset[i], cutoff=cutoff)
    return filtered_dataset




