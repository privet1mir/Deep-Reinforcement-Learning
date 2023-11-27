# Monte-Carlo Algorithm

In this project I introduce SARSA algorithm that can be used for solving deterministic environments, such as gym taxi, that we will explore this time. 

## Theory

$\varepsilon$-Greedy Policy:

$$\large{\pi(a|s) = \begin{cases}
  1 - \varepsilon + \varepsilon / m,& \text{ if } a \in \mathrm{argmax}_{a' \in \mathcal{A}}\, Q(s,a'), \\    
  \varepsilon / m,& \text{ otherwise } 
\end{cases}}$$

Let's $Q(s,a) = 0$ and $\varepsilon = 1$.

For each episode $k$ do:

While episode not done:

1. Being in state $S_t$ we do action $A_t \sim \pi(\cdot|S_t)$,
where $\pi = \varepsilon\text{-greedy}(Q)$, receive reward $R_t$, go to the state $S_{t+1}$, do action $A_{t+1} \sim \pi(\cdot|S_{t+1})$

2. On $(S_t,A_t,R_t,S_{t+1},A_{t+1})$ update $Q$:
   
$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_t + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))
$$

Reduce $\varepsilon$.

## Practice. Solve Taxi-v3

In this project I've implemented the Monte-Carlo algorithm to solve the [Taxi-v3](https://www.gymlibrary.dev/environments/toy_text/taxi/) environment from gym.

<img src="https://www.gymlibrary.dev/_images/taxi.gif" width="500">

## Results
