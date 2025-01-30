import numpy as np

from real_data.feature_selection.compute_features import compute_features
from real_data.feature_selection.check_points_features import check_features_are_standardized, plot_esd_for_feature, check_labels
from real_data.eval.fit_data import fit_data
from real_data.eval.eval_data import eval_esd_hessian
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
    if file_number == 1:
        z_real_values = np.linspace(0.001,0.1,60)
    elif file_number == 2 and alpha == 10:
        z_real_values = np.linspace(0.1,0.3,60)
    elif file_number == 2 and alpha == 20:
        z_real_values = np.linspace(0.1,0.3,60)
    elif file_number == 2 and alpha == 30:
        z_real_values = np.linspace(0.1,0.5,60)

    (X_train, y_train), (X_test, y_test) = compute_features('fashion_mnist',
                                                             feature_name, 
                                                             decorrelate=True, 
                                                             n_features=d, 
                                                             classes_to_keep=[2,4,6])   
    eval_esd_hessian_theory(X_train, y_train, X_test, y_test, 
                            file_number=file_number, seed=42, 
                            alpha=alpha, n_hidden=d, z_real_values=z_real_values)
    print("******logged esd theory!******")
