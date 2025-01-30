import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from multinomial_logistic.utils import batched_mlogit_jacobian




def fit_data(X_train_sampled, y_train_sampled, X_test, y_test,
             compute_esd=False, seed=42):
    """
    Learn the data using the given method.
    Assuming the desired samples are already selected.
    """

    np.random.seed(seed)
    logreg = LogisticRegression(           
                fit_intercept=False,
                penalty=None,
                solver='lbfgs',     
                max_iter=1000,
                random_state=seed
    )   

    logreg.fit(X_train_sampled, y_train_sampled)
    train_error = log_loss(y_train_sampled, logreg.predict_proba(X_train_sampled))
    test_error = log_loss(y_test, logreg.predict_proba(X_test))
    classification_error = 1 - accuracy_score(y_test, logreg.predict(X_test))

    Theta_hat = logreg.coef_
    Theta_hat = (Theta_hat - Theta_hat[0])[1:,:]
    results = {
        'train_error': train_error,
        'test_error': test_error,
        'classification_error': classification_error,
        'Theta_hat': Theta_hat,
        'esd_values': None
    }

    if compute_esd:
        esd = np.linalg.eigvals(_batched_hessian(Theta_hat, X_train_sampled))
        results['esd_values'] = esd

    return results



#####################
def _batched_hessian(theta, X):
    n, d = X.shape
    k = theta.shape[0]
    Beta_batch_hat = X @ theta.T
    scale = np.einsum('ni,nj->nij', X, X)
    Hessian = np.zeros((d*k, d*k))
    for i in range(n):
        Hessian += np.kron(batched_mlogit_jacobian(Beta_batch_hat)[i], scale[i])
    return Hessian/n