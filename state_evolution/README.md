State Evoltion Recursion for the Regularized Multinomial Logistic Regression.
The state evolution recursion is used to compute the fixed point of the state evolution equations.
The state evolution equations are:
$$
    S_{t+1} = 1/alpha * ((I - E[(I + S @ Jp(prox(g + yS; S)))^{-1}] + 2*lambda_reg*S_t))^{-1} @ S_t
    R_01_{t+1} = (I - alpha*2*lambda_reg * S_{t+1}) @ R_01_t - alpha * S_{t+1} @ E[(p(prox(g + yS; S)) - y)g_0.T] 
    schur_{t+1} = alpha * S_{t+1} @ E[(p(v)-y) @ (p(v)-y).T] @ S_{t+1}
$$

Note that the prox is computed at the S_t.
This recursion is used to compute the fixed point of the state evolution equations
$$
    (1/alpha - 1) * I + (2*lambda_reg) * S =  E[(I + S @ Jp(v))^{-1}] 
    (-2*lambda_reg) @ R_01 = E[(p(v) - y)g_0.T] 
        schur = alpha * S @ E[(p(v)-y) @ (p(v)-y).T] @ S
$$