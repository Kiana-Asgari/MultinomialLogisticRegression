from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
from scipy.linalg import sqrtm
from multinomial_logistic.MLE_empirical.utils.data_generation import generate_data
from multinomial_logistic.MLE_empirical.utils.test_error import mle_misclassification_test_error
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import test_error
from multinomial_logistic.utils import batched_mlogit_jacobian
"""
fitting the multinomial logistic regression model using skitlearn
if the data is not provided, it will generate data from N(0, I_d)
returns the test and train errors for each trial
"""




def fit_mle_skitlearn(alpha=None, k=None, d=None, n_trials=1, R_00=None,
                       X_train=None, y_train_onehot=None,
                       X_test=None, y_test_onehot=None, compute_eigenvalues=False,
                       learn_from_data=False, verbose=False):

    if learn_from_data:
        d = X_train.shape[1]
        k = y_train_onehot.shape[1]
        n_trials = 1



    Theta_0 = _initialize_Theta_0(d, k, R_00, learn_from_data)

    results = {
        'Theta_hats': np.zeros((n_trials, k, d)),
        'norms': np.zeros(n_trials),
        'test_errors': np.zeros(n_trials),
        'train_errors': np.zeros(n_trials),
        'misclass_test_errors': np.zeros(n_trials),
        'eigenvalues': np.zeros((n_trials, k*d)),
        'test_errors_skitlearn': np.zeros(n_trials),
        'misclass_test_errors_skitlearn': np.zeros(n_trials),
    }

    for i in range(n_trials):
        if not learn_from_data:
            X_train, y_train_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0, random_state=i)
        y_train = _concatenate_onehot(y_train_onehot)
        
        model = _fit_logistic_regression(X_train, y_train)
        Theta_hat = _get_Theta_hat(model)

        results['Theta_hats'][i] = Theta_hat
        results['norms'][i] = np.linalg.norm(Theta_hat - Theta_0)
        #results['test_errors'][i] = _compute_test_error(model, X_test, y_test_onehot, Theta_0=Theta_0, Theta_hat=Theta_hat, learn_from_data=learn_from_data)
        results['train_errors'][i] = log_loss(y_train, model.predict_proba(X_train))
        #results['misclass_test_errors'][i] = mle_misclassification_test_error(Theta_0=Theta_0, Theta_hat=Theta_hat)
        
        X_test, y_test_onehot = generate_data(alpha=1e3, d=d, k=k, Theta_0=Theta_0, random_state=11*i+i*i+1)
        y_test = _concatenate_onehot(y_test_onehot)
        results['test_errors'][i] = log_loss(y_test, model.predict_proba(X_test))
        results['misclass_test_errors'][i] = 1-accuracy_score(y_test, model.predict(X_test))
        #print('test error skitlearn:', results['test_errors_skitlearn'][i], 'test error empirical:', results['test_errors'][i])
        print('misclassification test error skitlearn:', results['misclass_test_errors'][i])




        if compute_eigenvalues:
            results['eigenvalues'][i] = np.linalg.eigvals(_batched_hessian(Theta_hat, X_train))
        if verbose:
            print(f'    ***      trial {i} done')    
            print(f'    ***          test error: {results["test_errors"][i]}')
            print(f'    ***          train error: {results["train_errors"][i]}')
            print(f'    ***          misclassification test error: {results["misclass_test_errors"][i]}')
    if verbose:
        print(f'    *** all trials done for alpha: {alpha}, k: {k}, d: {d}, n_trials: {n_trials}')   
        print(f'    ***         mean test error: {np.mean(results["test_errors"])}')
        print(f'    ***         mean train error: {np.mean(results["train_errors"])}')
        print(f'    ***         mean misclassification test error: {np.mean(results["misclass_test_errors"])}')
    return results




###################################################################################
# helper functions for skitlearn
###################################################################################

def _compute_test_error(model, X_test, y_test_onehot, Theta_0, Theta_hat, learn_from_data):
    """Compute test error based on available test data."""
    if learn_from_data:
        y_test = _concatenate_onehot(y_test_onehot)
        return log_loss(y_test, model.predict_proba(X_test))
    else:
        return test_error(Theta_0, Theta_hat)


def _initialize_Theta_0(d, k, R_00, learn_from_data):
    zeros_pad = np.zeros((k, d-k))  # k x (d-k) matrix of zeros
    Theta_0 = np.hstack([sqrtm( R_00), zeros_pad]) if not learn_from_data else np.zeros((k, d)) # concatenate horizontally to get k x d matrix
    return Theta_0


def _get_Theta_hat(model):
    Theta_hat = model.coef_
    Theta_hat = (Theta_hat - Theta_hat[0])[1:,:]   # remove baseline class
    return Theta_hat


def _fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(penalty=None,
                                fit_intercept=False,
                                max_iter=500,
                                solver='lbfgs')
    model.fit(X_train, y_train)
    return model



def _concatenate_onehot(y_onehot):
    y = np.argmax(y_onehot, axis=1) + np.max(y_onehot, axis=1)
    return y



def _batched_hessian(theta, X):
    n, d = X.shape
    k = theta.shape[0]
    Beta_batch_hat = X @ theta.T
    scale = np.einsum('ni,nj->nij', X, X)
    Hessian = np.zeros((d*k, d*k))
    for i in range(n):
        Hessian += np.kron(batched_mlogit_jacobian(Beta_batch_hat)[i], scale[i])
    return Hessian/n



######################################


