"""
Initialize the multinomial_logistic package.
Import commonly used libraries.
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.special import softmax, logsumexp, expit
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cubature import cubature

# Make imported modules and functions available throughout the package
__all__ = [
    'np',
    'fsolve',
    'minimize',
    'softmax',
    'logsumexp',
    'expit',
    'nquad',
    'multivariate_normal',
    'sqrtm',
    'cubature'
] 