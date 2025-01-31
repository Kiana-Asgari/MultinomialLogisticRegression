import numpy as np

from real_data.feature_selection.compute_features import compute_features
from real_data.feature_selection.check_points_features import check_features_are_standardized, plot_esd_for_feature, check_labels
from real_data.eval.fit_data import fit_data
from real_data.eval.eval_data import eval_esd_hessian
from real_data.eval.eval_theory import eval_error_theory
from real_data.plotting.plot_errors import plot_errors_comparison

def test_compute_features():
    print("Testing compute_features...")
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist', 'tanh', decorrelate=True, n_features=500, classes_to_keep=[2,4,6])
    print("compute_features passed!")
    print("Testing check_features_are_standardized...")
    check_features_are_standardized(X_train)
    print("check_features_are_standardized passed!")  
    print("Testing esd of features...")
    plot_esd_for_feature(X_train)
    print("esd of features passed!(check the log folder)")  
    print("Testing labels...")
    check_labels(y_train, y_test, X_train, X_test, classes_to_keep=[2,4,6])
    print("labels passed!")
    print("******All tests passed for compute features from fashion mnist!******")



def test_fitting_whole_dataset():
    print("Testing fitting the whole dataset...")
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             'tanh', 
                                                             decorrelate=True, 
                                                             n_features=500, 
                                                             classes_to_keep=[2,4,6])
    results = fit_data(X_train, y_train, X_test, y_test, compute_esd=False)
    print("fitting the whole dataset passed!")
    print('train error: ', results['train_error'])
    print('test error: ', results['test_error'])
    print('classification error: ', results['classification_error'])
    R_00 = results['Theta_hat']@results['Theta_hat'].T
    print('R_00: ', R_00)
    print("******All tests passed for fitting the whole fshion mnist dataset!******")

from real_data.eval.eval_data import eval_data
def test_fitting_sampled_dataset():
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             'tanh', 
                                                             decorrelate=True, 
                                                             n_features=500, 
                                                             classes_to_keep=[2,4,6])   
    for alpha in [6,7,8,9]:        
        print("Testing fitting the sampled dataset for alpha = ", alpha, "...")
        test_errors, train_errors, classification_errors = eval_data(X_train,
                                                                      y_train,   
                                                                      X_test, 
                                                                      y_test, 
                                                                      alpha=alpha, 
                                                                      n_iter=50)
        print("     test errors: ", np.mean(test_errors), 'std: ', np.std(test_errors))
        print("     train errors: ", np.mean(train_errors), 'std: ', np.std(train_errors))
        print("     classification errors: ", np.mean(classification_errors), 'std: ', np.std(classification_errors))
    print("******All tests passed for fitting the sampled fshion mnist dataset!******")



def test_eval_esd_hessian():
    print("Testing eval_esd_hessian for alpha = 20...")
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             'tanh', 
                                                             decorrelate=True, 
                                                             n_features=350, 
                                                             classes_to_keep=[2,4,6])   
    eval_esd_hessian(X_train, y_train, X_test, y_test, alpha=20, n_iter=5)
    print("******All tests passed for eval_esd_hessian!******")


from real_data.eval.eval_theory import eval_esd_hessian_theory
def test_eval_esd_hessian_theory():
    print("Testing eval_esd_hessian_theory for alpha = 20...")
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             'tanh', 
                                                             decorrelate=True, 
                                                             n_features=350, 
                                                             classes_to_keep=[2,4,6])   
    density_list = eval_esd_hessian_theory(X_train, y_train, X_test, y_test, alpha=20)
    print("density_list: ", density_list)
    print("******All tests passed for eval_esd_hessian_theory!******")

def log_esd_theory(file_number, alpha):
    d=350
    feature_name='tanh'
    if alpha == 10:
        z_real_values = np.linspace(0.001,0.35,100)
    elif alpha == 20:
        z_real_values = np.linspace(0.001,0.35,100)
    elif alpha == 30:
        z_real_values = np.linspace(0.001,0.55,100)

    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             feature_name, 
                                                             decorrelate=True, 
                                                             n_features=d, 
                                                             classes_to_keep=[2,4,6])   
    eval_esd_hessian_theory(X_train, y_train, X_test, y_test, 
                            file_number=file_number, seed=42,  feature_name=feature_name,
                            alpha=alpha, n_hidden=d, z_real_values=z_real_values)
    print("******logged esd theory!******")



def log_error_theory(file_number, n_hidden, feature_name):
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             feature_name, 
                                                             decorrelate=True, 
                                                             n_features=n_hidden, 
                                                             classes_to_keep=[2,4,6])   
    eval_error_theory(X_train, y_train, X_test, y_test,
                      feature_name=feature_name, n_hidden=n_hidden, 
                      file_number=file_number, seed=42)
    print("******logged error theory!******")



from real_data.eval.eval_data import eval_errors_empirical
def log_error_empirical(n_hidden, feature_name='tanh', file_number=1):
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             feature_name, 
                                                             decorrelate=True, 
                                                             n_features=n_hidden, 
                                                             classes_to_keep=[2,4,6])   
    eval_errors_empirical(X_train, y_train, X_test, y_test, n_iter=50,
                          feature_name=feature_name, n_hidden=n_hidden, 
                          file_number=file_number, seed=42)
    print("******logged error empirical!******")

from real_data.eval.eval_data import eval_bayesian_error
def log_bayesian_error(n_hidden, feature_name='tanh', file_number=1):
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             feature_name, 
                                                             decorrelate=True, 
                                                             n_features=n_hidden, 
                                                             classes_to_keep=[2,4,6])   
    eval_bayesian_error(X_train, y_train, X_test, y_test, feature_name, n_hidden, file_number=file_number, seed=42)
    print("******logged bayesian error!******")

from real_data.eval.eval_theory import eval_irreducible_error_theory
def log_irreducible_error(n_hidden, feature_name='tanh', file_number=1):
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             feature_name, 
                                                             decorrelate=True, 
                                                             n_features=n_hidden, 
                                                             classes_to_keep=[2,4,6])   
    eval_irreducible_error_theory(X_train, y_train, X_test, y_test, feature_name, n_hidden, file_number=file_number, seed=42)
    print("******logged irreducible error!******")


from real_data.plotting.plot_esd import plot_esd_density
def plots(alpha, file_number=1):
    """
    Reads the ESD Hessian data for a given alpha and plots density vs z_real values.
    
    Args:
        alpha (float): The alpha value to plot
        file_number (int): The file number suffix (default=1)
    """
    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             'tanh', 
                                                             decorrelate=True, 
                                                             n_features=350, 
                                                             classes_to_keep=[2,4,6])
    plot_esd_density(X_train, y_train, X_test, y_test, alpha, file_number)

from real_data.plotting.plot_bays import plot_bays
def plot_errors(n_hidden, feature_name='tanh', file_number=1):
    plot_errors_comparison(feature_name, n_hidden, file_number)
    print("******plot errors comparison passed!******")
    plot_bays(feature_name, n_hidden, file_number)
    print("******plot bayesian errors passed!******")
