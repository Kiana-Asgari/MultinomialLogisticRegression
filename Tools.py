def unwrap(vars, k, k_0): # unwrap (S, R_10, R_11).flatten()
    vars = np.array(vars)
    assert len(vars) == k*k + k*k_0 + k*k
    S = vars[:k*k].reshape((k, k))
    R_10 = vars[k*k:k*k + k*k_0].reshape((k, k_0))
    R_11 = vars[k*k + k*k_0:].reshape((k, k))
    return S, R_10, R_11

def wrapper(S, R_10, R_11): # wrap (S, R_10, R_11)
    return np.concatenate((S.flatten(), R_10.flatten(), R_11.flatten()))

def mlogit(beta): # returns a k+1 dimentional logistic perobablity vector with the prob of 0 as the last element
    return softmax(np.append(beta,0))

def schur_complement(R_10, R_11, R_00): # Returns R\R_00
    R_00inv = np.linalg.inv(R_00)
    return R_11 - R_10 @ R_00inv @ (R_10).T

def normal_basis(i, k): # Reutrns the standard basis of R^k
    assert i in range(-1, k)
    return np.zeros(k) if i == -1 else np.eye(k)[:, i]
