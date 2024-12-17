# MultinomialLogisticRegression

The state evolution equation for R_01:

$R_{01}^{t+1} = (I - \alpha \cdot 2\lambda_{reg} \cdot S^{t+1}) \cdot R_{01}^t - \alpha \cdot S^{t+1} \cdot \mathbb{E}[(p(\text{prox}(g + yS; S)) - y)g_0^T]$