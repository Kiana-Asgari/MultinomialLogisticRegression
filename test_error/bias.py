"""
plots the different values of R_01 for different values of lambda_reg, for k=1, k_0=1, R_00=I   
"""

import numpy as np
from state_evolution.full_recursion import state_evolution_full_recursion
from test_error.utils import plot_array


def plot_lambda_vs_bias(R_00, lambda_reg_min, lambda_reg_max, alpha, k, k_0):
    lambda_reg_values = np.linspace(lambda_reg_min, lambda_reg_max, 50)
    R_01_values = []
    for lambda_reg in lambda_reg_values:
        schur, R_01, S = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
        R_01_values.append(R_01)
    nclass = k+1
    title = f"R_01 vs lambda_reg for alpha={alpha:.2f}, number of class={nclass:d}"
    name = f"R_01_vs_lambda_reg_alpha={alpha:.2f}_nclass={nclass:d}"    
    plot_array(lambda_reg_values, R_01_values, title=title,\
               x_label="lambda_reg", y_label="R_01", name=name)
    return lambda_reg_values, R_01_values


def plot_alpha_vs_bias(R_00, alpha_min, alpha_max, lambda_reg, k, k_0):
    alpha_values = np.linspace(alpha_min, alpha_max, 50)
    R_01_values = []
    for alpha in alpha_values:
        schur, R_01, S = state_evolution_full_recursion(R_00=R_00, schur_0=R_00, R_01_0=np.zeros((k_0,k)),\
                                        lambda_reg=lambda_reg, alpha=alpha, k=k, k_0=k_0)
        R_01_values.append(R_01)
    nclass = k+1
    title = f"R_01 vs alpha for lambda={lambda_reg:.2f}, number of class={nclass:d}"
    name = f"R_01_vs_alpha_lambda={lambda_reg:.2f}_nclass={nclass:d}"
    plot_array(alpha_values, R_01_values, title=title,\
               x_label="alpha", y_label="R_01", name=name)
    return alpha_values, R_01_values

