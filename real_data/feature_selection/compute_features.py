from real_data.mnist_data.process_data import get_cleaned_fashion_mnist_data
from real_data.feature_selection.rf import feature_selection

def compute_features(dataset_name='fashion_mnist',
                     method='relu',
                     decorrelate=True,
                     n_features=500,
                     classes_to_keep=[2,4,6]):
    """
    Load the data from the dataset_name. so far only fashion_mnist is supported.
    Compute the features using the rf+activation method,
    and decorrelate if desired.
    """

    if dataset_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = get_cleaned_fashion_mnist_data(classes_to_keep=classes_to_keep)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    X_train, X_test = feature_selection(X_train=X_train,
                                         X_test=X_test,
                                         n_features=n_features,
                                         method=method,
                                         decorrelate=decorrelate)

    return (X_train, y_train), (X_test, y_test)