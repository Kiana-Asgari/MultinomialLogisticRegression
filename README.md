# MultinomialLogisticRegression

The state evolution equation for $R_{01}$:

State Evolution Recursion for the Regularized Multinomial Logistic Regression.
The state evolution recursion is used to compute the fixed point of the state evolution equations.
The state evolution equations are:

$$S_{t+1} = \frac{1}{\alpha} \left(I - \mathbb{E}\left[\left(I + S \cdot \text{Jp}(v_t)\right)^{-1}\right] + 2\lambda_{\text{reg}}S_t\right)^{-1} \cdot S_t$$

$$R_{01_{t+1}} = \left(I - \alpha \cdot 2\lambda_{\text{reg}} \cdot S_{t+1}\right) \cdot R_{01_t} - \alpha \cdot S_{t+1} \cdot \mathbb{E}\left[\left(p(\text{prox}(g + yS; S)) - y\right)g_0^T\right]$$

$${R/R_{00}}_{t+1} = \alpha  S_{t+1}  \mathbb{E}
\left[(p(v_t)-y) \cdot (p(v_t)-y)^T\right]  S_{t+1}$$

where $v_t = \text{prox}(g_t + yS_t; S_t)$ for $g_t,g_0\sim N(0,R_t)$ and $y\sim p(g_0)$

This recursion is used to compute the fixed point of the state evolution equations:

$$\left(\frac{1}{\alpha} - 1\right) \cdot I + (2\lambda_{\text{reg}}) \cdot S = \mathbb{E}\left[\left(I + S \cdot \text{Jp}(v)\right)^{-1}\right]$$

$$(-2\lambda_{\text{reg}}) \cdot R_{01} = \mathbb{E}\left[(p(v) - y)g_0^T\right]$$

$$\text{R/R_{00}} = \alpha \cdot S \cdot \mathbb{E}\left[(p(v)-y) \cdot (p(v)-y)^T\right] \cdot S$$