"""
Calulating the integrands inside the integral of the fixed point system.
"""

def fixed_point_integrands(Z, S, A, R_00, schur_root, alpha, k, k_0): # Z ~ N(0,I_k)

    g, g_0 = coloring_transform(Z, A, R_00, schur_root, alpha, k, k_0) # (g,g_0) ~ N(0, R)
    prox_g = prox(S, g)
    prob_y = mlogit(g_0)
    p = mlogit(prox_g)[:-1]

    integrand1, integrand2, integrand3 = np.zeros((k, k)), np.zeros((k, k)), np.zeros((k, k))

    for i in range(-1, k):
        y = normal_basis(i, k)
        p = mlogit(prox_g + S @ y)[:-1]
        integrand1 += np.outer(p - y, p - y) * prob_y[i]
        integrand2 += np.outer(p - y, g_0) * prob_y[i]
        integrand3 += np.outer(p - y, prox_g) * prob_y[i]

    mvn = multivariate_normal(mean=np.zeros(k+k_0), cov=np.eye(k+k_0))
    return [integrand1, integrand2, integrand3] * mvn.pdf(Z)
