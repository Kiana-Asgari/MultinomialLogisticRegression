import numpy as np
import cupy as cp
from cuml.linear_model import LogisticRegression as cuLogisticRegression  # type: ignore
from cuml.internals.memory_utils import set_global_output_type  # type: ignore
from scipy.linalg import sqrtm
from multinomial_logistic.MLE_empirical.utils.data_generation import generate_data
from multinomial_logistic.utils import batched_mlogit_jacobian


set_global_output_type('cupy')
_CUDA_DEVICE = cp.cuda.Device(3)
_free_mem, _total_mem = cp.cuda.runtime.memGetInfo()
_memory_limit = int(_total_mem * 0.7)
_memory_pool = cp.cuda.MemoryPool()
_memory_pool.set_limit(_memory_limit)
cp.cuda.set_allocator(_memory_pool.malloc)

"""
fitting the multinomial logistic regression model using skitlearn
if the data is not provided, it will generate data from N(0, I_d)
returns the test and train errors for each trial
"""


def fit_mle_skitlearn(alpha=None, k=None, d=None, n_trials=100, R_00=None,
                       X_train=None, y_train_onehot=None,
                       X_test=None, y_test_onehot=None, compute_eigenvalues=False,
                       learn_from_data=False, verbose=False, seed=42):

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
    np.random.seed(seed)
    seeds = np.random.randint(0, 1000000, 1000)
    Theta_0_gpu = cp.asarray(Theta_0, dtype=cp.float64)

    for i in range(n_trials):
        iter = 2*i
        if not learn_from_data:
            X_train, y_train_onehot = generate_data(alpha=alpha, d=d, k=k, Theta_0=Theta_0, random_state=seeds[iter])
        y_train = _concatenate_onehot(y_train_onehot)
        
        X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train, dtype=cp.int32)

        model = _fit_logistic_regression(X_train_gpu, y_train_gpu, seed=seed)
        Theta_hat_gpu = _get_Theta_hat(model)
        Theta_hat_cpu = cp.asnumpy(Theta_hat_gpu)

        results['Theta_hats'][i] = Theta_hat_cpu
        results['norms'][i] = float(cp.linalg.norm(Theta_hat_gpu - Theta_0_gpu).item())

        train_proba = cp.asarray(model.predict_proba(X_train_gpu))
        results['train_errors'][i] = _log_loss_gpu(y_train_gpu, train_proba)
        
        X_test, y_test_onehot = generate_data(alpha=10, d=d, k=k, Theta_0=Theta_0, random_state=seeds[iter+1])
        y_test = _concatenate_onehot(y_test_onehot)
        X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
        y_test_gpu = cp.asarray(y_test, dtype=cp.int32)

        test_proba = cp.asarray(model.predict_proba(X_test_gpu))
        results['test_errors'][i] = _log_loss_gpu(y_test_gpu, test_proba)
        preds = cp.asarray(model.predict(X_test_gpu), dtype=cp.int32)
        results['misclass_test_errors'][i] = 1.0 - _accuracy_gpu(y_test_gpu, preds)
        print(f'iter: {i}, test error: {results["test_errors"][i]:.3f}, train error: {results["train_errors"][i]:.3f}, misclassification: {results["misclass_test_errors"][i]:.3f}')


        if compute_eigenvalues:
            results['eigenvalues'][i] = np.linalg.eigvals(_batched_hessian(Theta_hat_cpu, X_train))
            print('iter: ', iter, 'eigenvalues: ', results['eigenvalues'][i])

    gpu_clear_cache()
    return results




###################################################################################
# helper functions for skitlearn
###################################################################################

def _initialize_Theta_0(d, k, R_00, learn_from_data):
    zeros_pad = np.zeros((k, d-k))  # k x (d-k) matrix of zeros
    Theta_0 = np.hstack([sqrtm( R_00), zeros_pad]) if not learn_from_data else np.zeros((k, d)) # concatenate horizontally to get k x d matrix
    return Theta_0


def _get_Theta_hat(model):
    Theta_hat = cp.asarray(model.coef_, dtype=cp.float32)
    Theta_hat = (Theta_hat - Theta_hat[0])[1:, :]
    return Theta_hat


def _fit_logistic_regression(X_train_gpu, y_train_gpu, seed=42):
    model = cuLogisticRegression(
        penalty=None,
        fit_intercept=False,
        max_iter=1500,
        tol=1e-6,
        C=1.0,
        solver='qn',
        verbose=0
    )
    model.fit(X_train_gpu, y_train_gpu)
    return model



def _concatenate_onehot(y_onehot):
    y = np.argmax(y_onehot, axis=1) + np.max(y_onehot, axis=1)
    return y


def _log_loss_gpu(y_true_gpu, probs_gpu):
    y_true_gpu = y_true_gpu.astype(cp.int32)
    probs_gpu = probs_gpu.astype(cp.float64)
    eps = cp.finfo(probs_gpu.dtype).eps
    probs_gpu = cp.clip(probs_gpu, eps, 1 - eps)
    losses = -cp.log(probs_gpu[cp.arange(y_true_gpu.shape[0]), y_true_gpu])
    return float(cp.mean(losses).item())


def _accuracy_gpu(y_true_gpu, y_pred_gpu):
    y_true_gpu = y_true_gpu.astype(cp.int32)
    y_pred_gpu = y_pred_gpu.astype(cp.int32)
    accuracy = cp.mean((y_true_gpu == y_pred_gpu).astype(cp.float32))
    return float(accuracy.item())



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


#######################################################################
## memory clean up
#######################################################################
import torch, gc

def gpu_barrier(device=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

def gpu_clear_cache(device=None):
    gpu_barrier(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def gpu_reset_memstats(device=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)