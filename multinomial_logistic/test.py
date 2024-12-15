import numpy as np
from multinomial_logistic.utils import batched_mlogit
from scipy.stats import multivariate_normal
from multinomial_logistic.integration import quadrature_integration, coloring_transform
from scipy.linalg import sqrtm
from multinomial_logistic.prox import prox_fp_iteration
from multinomial_logistic.fixed_point_system.fp_integrands import batched_fixed_point_integrands
from multinomial_logistic.fixed_point_system.fp_system import fixed_point_system
from multinomial_logistic.utils import wrapper
from multinomial_logistic.fixed_point_system.fp_solver import fixed_point_solver_iterative, fixed_point_solver_newton

def mock_func_prox(Z_batch, S, A, R_00, schur_root, alpha, k, k_0):
    mvn = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0))
    g_batch, g_0_batch = coloring_transform(Z_batch.T, A, R_00, schur_root, alpha, k, k_0)
    #CALL PROXIMAL FIXED POINT ITERATION TO TEST THE SPEED
    prox = prox_fp_iteration(g_batch, S)
    outer_products = np.einsum('ij,ik->ijk', g_batch, g_batch)  # Shape (n_samples, k+k_0, k+k_0)
    # Flatten the outer product for each sample in the batch
    outer_products_flattened = outer_products.reshape(Z_batch.shape[0], -1)  # Shape (n_samples, (k+k_0)^2)
    # Compute the PDF for all samples in Z_batch
    pdf_values = mvn.pdf(Z_batch)  # Shape (n_samples,) 
    # Perform element-wise multiplication (broadcasting the PDF across the flattened outer products)
    result = outer_products_flattened * pdf_values[:, np.newaxis]  # Shape (n_samples, (k+k_0)^2)
    return result

# integral test
def mock_func(Z, S, A, R_00, schur_root, alpha, k, k_0):
    mvn = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0))
    g_all, g_0_all = coloring_transform(Z.T, A, R_00, schur_root, alpha, k, k_0)
    outer_products = np.einsum('ij,ik->ijk', g_all, g_all)  # Shape (n_samples, k+k_0, k+k_0)
    # Flatten the outer product for each sample in the batch
    outer_products_flattened = outer_products.reshape(Z.shape[0], -1)  # Shape (n_samples, (k+k_0)^2)
    # Compute the PDF for all samples in Z_batch
    pdf_values = mvn.pdf(Z)  # Shape (n_samples,) 
    # Perform element-wise multiplication (broadcasting the PDF across the flattened outer products)
    result = outer_products_flattened * pdf_values[:, np.newaxis]  # Shape (n_samples, (k+k_0)^2)
    return result

def test_integration_1():
    print("Running test_integration 1")
    k, k_0 = 1, 1
    alpha = 1.0
    S = np.eye(k)
    R_11 = np.array([[10]])
    R_10 = np.array([[1]])

    R_00 = np.array([[1]])
    A = R_10 @ np.linalg.inv(sqrtm(R_00))
    schur_root = sqrtm(schur_complement(R_10, R_11, R_00))

    result = quadrature_integration(mock_func, S, A, R_00, schur_root, alpha, k, k_0)
    print('the integration result is ', result)

# Define the test function
def test_integration_2():
    # Example parameters
    print("Running test_integration with prox")
    k, k_0 = 2, 2
    alpha = 1.0
    S = np.eye(k)
    R_11 = np.array([
    [1.9383451 , 0.76780201],
    [0.76780201, 0.80231093]
        ])
    R_10 = np.array([
    [1.4940289 , 0.75654844],
    [1.04554899, 0.33242197]
    ]   )

    R_00 = np.array([
    [1.80385488, 0.83237389],
    [0.83237389, 0.80474618]
    ])

    A = R_10 @ np.linalg.inv(sqrtm(R_00))
    schur_root = sqrtm(schur_complement(R_10, R_11, R_00))

    result = quadrature_integration(mock_func_prox, S, A, R_00, schur_root, alpha, k, k_0)
    print('the integration result is ', result)

def test_prox():
    print("Running test_prox")
    N, k = 5, 2  # Example dimensions
    g_batch = np.array([[1, 0], [1, 1], [-1, 1], [0, 0], [1, 0]])  # Batch of N vectors, each of size k
    S = np.eye(k)  # Shared matrix S (k x k)

    beta_solutions = prox_fp_iteration(g_batch, S)

    print('beta solutions', beta_solutions)


def test_fp_integrands():
    print("Running test_fp_integrands")
    N, k, k_0 = 5, 2, 2  # Example dimensions
    Z_batch = np.ones((N, k+k_0))
    S = np.eye(k)  # Shared matrix S (k x k)
    A = np.eye(k)
    R_00 = np.eye(k)
    schur_root = np.eye(k)
    alpha = 1.0
    result = batched_fixed_point_integrands(Z_batch, S, A, R_00, schur_root, alpha, k, k_0)
    print('result', result.shape, result)

def test_fp_system():
    print("Running test_fp_system")
    k, k_0 = 2, 2  # Example dimensions
    vars = wrapper(10 * np.eye(k), np.eye(k), np.eye(k))
    R_00 = np.eye(k)
    alpha = 1.0
    result = fixed_point_system(vars, R_00, alpha, k, k_0)
    print('result', result.shape, result)

def test_fp_solver():
    print("Running test_fp_solver")
    k, k_0 = 2, 2
    R_00 = np.eye(k)
    alpha = 10
    #S, R_10, R_11 = fixed_point_solver_iterative(R_00, alpha, k, k_0)
    #print('S, R_10, R_11', S, R_10, R_11)
    S, R = fixed_point_solver_newton(R_00, alpha, k, k_0)
    print('S, R', S, R)