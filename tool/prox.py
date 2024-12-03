"""
Multivariate Proximal Oprator
First approach: calculated through KKT
"""

def prox_deriv(beta, g, S):
    return (beta - g) + (S @ mlogit(beta)[:, :-1].T).T

def prox(S, g):
    prox_initial_guess = np.zeros_like(g)
    prox = fsolve(prox_deriv, prox_initial_guess, args=(g, S))
    return prox

"""
Multivariate Proximal Oprator
second approach: using built in minimizer
"""
def smoothed_mlogit(beta, S, g):
    S_inv = np.linalg.inv(S)
    return 0.5 *  ((beta - g) @ S_inv @ (beta - g)) + logsumexp(np.append(beta, 0))

def prox_minimization_app(S, g):
    initial_beta = np.zeros(len(g))
    return minimize(smoothed_mlogit, initial_beta, args=(S, g)).x
