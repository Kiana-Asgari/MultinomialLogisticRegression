import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.linalg import sqrtm
from multinomial_logistic.evaluation.utils import plot_array
from state_evolution.full_recursion import state_evolution_full_recursion
from sklearn.exceptions import ConvergenceWarning
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from multinomial_logistic.MLE_empirical.mle_empirical_baseline import fit_mle_baseline


def plot_mle_vs_lambda_reg(lambda_reg_min, lambda_reg_max, R_00,k, save_path, d = 800, n_trials=100):

    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, 1, endpoint=False)
    alpha_values = [2]

    norm_batches = np.zeros((len(alpha_values), len(lambda_reg_values)))
    
    # Create theta_0 by concatenating R_00 with zeros


    legends = [ 'alpha=4', 'alpha=2']


    for i, alpha in enumerate(alpha_values):
        for j, lambda_reg in enumerate(lambda_reg_values):
            schur, R_01, S, _ = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k,k)),\
                                            lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k)
            R_11 = schur + R_01 @ np.linalg.inv(R_00) @ R_01.T

            
            theta_hat, avg_norm = fit_mle_baseline(alpha=alpha, k=k, lambda_reg=2 *lambda_reg,\
                                                    R_00 = R_00, d=d, n_trials=n_trials)
            #theta_test, avg_norm_test = fit_mle(alpha=alpha, k=k, lambda_reg=2 * lambda_reg,\
            #                                        Theta_0=theta_0, d=d, n_trials=n_trials)
            norm_batches[i,j] = avg_norm

            #print(f'     gram_matrix scikit: {gram_matrix_test}')
            print(f'     avg_distance baseline: {avg_norm}')
            #print(f'     avg_distance scikit: {avg_norm_test}')
            #print(f'     avg_theta difference: {np.linalg.norm(theta_hat - theta_test)**2}')
            print(f'     R_01: {R_01}')
            print(f'     R_11: {R_11}')
            print(f'     theoretical distance: {np.trace(R_00) + np.trace(R_11) - np.trace(R_01) - np.trace(R_01.T)}')

    title = f"empirical ||theta-theta_0||_F vs lambda_reg, number of class={k+1:d},R_00= ({R_00})"
    name = f"empirical_F_norm_vs_lambda_reg_nclass={k+1:d}_R_00={R_00}" 
    plot_array(2* lambda_reg_values, norm_batches, legends=legends,\
                title=title,\
                x_label="lambda_reg", y_label="F_norm",\
                name=name, save_path=save_path)










###############################################################################

def fit_mle(alpha, k, lambda_reg, Theta_0, d, n_trials):
    avg_Theta_hat = np.zeros((k, d))
    avg_norm = 0
    print('lambda_reg: ', lambda_reg)

    warnings.filterwarnings('ignore')
    if k == 1:
        model =  LogisticRegression(
            penalty='l2',
            C=2/lambda_reg,
            fit_intercept=False,  # so it aligns with the "baseline" idea
            solver='saga'
        )
    else:
        model = LogisticRegression(
            multi_class='multinomial', 
            penalty='l2',
            C=2/(lambda_reg),
            fit_intercept=False,  # so it aligns with the "baseline" idea
            solver='saga'
        )


    for i in range(n_trials):
        X, Y_onehot = generate_data(alpha, d, k, Theta_0, random_state=i)
        y_labels = one_hot_to_integers(Y_onehot)  # shape (n,)


        model.fit(X, y_labels)

        Theta_hat = model.coef_

        if Theta_hat.shape[0] == k+1:
            Theta_hat = (Theta_hat - Theta_hat[0])[-k:,:]  # shape (k, d) normalized

        avg_norm += np.linalg.norm(Theta_0- Theta_hat)**2
        avg_Theta_hat += Theta_hat
        #print(' avg norm with reg:', np.linalg.norm(theta_0- Theta_hat)**2)



    avg_Theta_hat /= n_trials
    avg_norm /= n_trials

    return avg_Theta_hat, avg_norm








###############################################################################



def generate_data(alpha, d, k, theta_0, random_state=0):
    """
    Generate n data points (X_i, Y_i).
    
    - X_i ~ N(0, I_d)
    - Y_i in {0_k (all zero), e_1, ..., e_k} with probabilities:
        P(Y_i = e_j) = exp(x_i^T theta_j) / [1 + sum_{l=1}^k exp(x_i^T theta_l)]
        P(Y_i = 0_k) = 1 / [1 + sum_{l=1}^k exp(x_i^T theta_l)]
    
    :param n: number of samples
    :param d: dimension of each feature vector
    :param k: number of non-baseline classes (so total classes = k+1)
    :param theta_0: true parameters, shape = (k, d)
    :param random_state: for reproducibility
    :return: X (n,d), Y_onehot (n,k)  [one-hot for classes 1..k, 0_k for baseline]
    """
    np.random.seed(random_state)
    n = np.ceil(alpha*d).astype(int)
    X = np.random.randn(n, d)  # shape (n, d)
    
    # Logits for non-baseline classes: shape (n, k)
    logits = X @ theta_0.T
    exp_logits = np.exp(logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    
    # Prob(baseline) = 1 / (1 + sum_{j=1}^k exp(x theta_j))
    p0 = 1.0 / (1.0 + sum_exp)
    p0 = p0.flatten()  # Convert from (n,1) to (n,)
    
    # Prob(class j) for j=1..k
    pj = exp_logits / (1.0 + sum_exp)
    
    # Draw labels
    Y_onehot = np.zeros((n, k))  # shape (n, k)
    for i in range(n):
        # pick from among k+1 classes
        # class 0 with prob p0[i], or class j in 1..k with prob pj[i, j-1]
        choices = np.arange(k+1)           # 0..k
        probs = np.concatenate([[p0[i]], pj[i]])
        cls = np.random.choice(choices, p=probs)
        if cls > 0:
            Y_onehot[i, cls-1] = 1.0   # make it e_{cls} in 1..k
    return X, Y_onehot

def one_hot_to_integers(Y_onehot):
    """
    Convert one-hot representation (including baseline all-zeros) 
    to integer labels 0..k.
    
    If row is [0,0,...,0], => label 0 (baseline).
    If row is e_j => label j (which is j in {1..k}).
    """
    labels = []
    for row in Y_onehot:
        if np.all(row == 0):
            labels.append(0)
        else:
            j = np.argmax(row) + 1  # e_j => label j
            labels.append(j)
    return np.array(labels)






def mle_test(): 
    # Setup
    n = 5000      # number of data points
    d = 5         # feature dimension
    k = 1         # non-baseline classes => total classes = 4
    np.random.seed(1234)
    
    # True parameter: shape = (k, d)
    #   In a baseline formulation, class 0 is fixed at 0_d, 
    #   class j in {1..k} has vector theta_0[j-1, :].
    theta_0 = np.array([
        [ 1.0,  0.0,  0.0,  0.0, 0.0]
    ])
    
    # 1) Generate data
    X, Y_onehot = generate_data(n, d, k, theta_0, random_state=42)
    
    # 2) Convert from one-hot (including baseline) => integer labels
    #    Classes: 0 (baseline), 1..k
    y_labels = one_hot_to_integers(Y_onehot)  # shape (n,)
    
    # 3) Fit scikit-learn logistic regression in "multinomial" mode, 
    #    with no regularization, and no intercept.
    #    This performs MLE => Minimizes negative log-likelihood.
    logreg = LogisticRegression(
        #multi_class='multinomial',
        penalty=None,
        fit_intercept=False,
        solver='lbfgs',
        max_iter=500
    )
    logreg.fit(X, y_labels)
    
    # scikit-learn will learn coefficients for each of the (k+1) classes.
    # shape => (k+1, d)
    Theta_hat = logreg.coef_
    print(f"Theta_hat: {Theta_hat}")
    Theta_hat = (Theta_hat - Theta_hat[0])[-k:,:]  # shape (k+1, d)
    
    print("True parameter theta_0 (baseline parameterization), shape = (k, d):")
    print(theta_0)
    print("\nLearned scikit-learn coefficients, shape = (k+1, d):")
    print(Theta_hat)
    print("\nExplanation:")
    print("  • The first row in Theta_hat is scikit-learn's parameters for class 0.")
    print("  • The next k rows correspond to classes 1..k.")
    print("  • Because scikit-learn does NOT force class 0 to be identically zero,")
    print("    these can differ from theta_0 by a constant shift across rows.")
    

