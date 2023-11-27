# SARSA Algorithm

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/SARSA%20Algorithm.%20Taxi-v3/SARSA_algorithm.ipynb) I introduce SARSA algorithm that can be used for solving deterministic environments, such as gym taxi, that we will explore this time. 

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

After implementing such algorithm we can see that it is perfectly solve the environment. Also we can compare it with two similar algos: Monte-Carlo & Q-Learning. 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/SARSA%20Algorithm.%20Taxi-v3/images/sarsa.png" width="800">

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/SARSA%20Algorithm.%20Taxi-v3/images/comparison_graph1.png" width="800">

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/SARSA%20Algorithm.%20Taxi-v3/images/comparison_graph2.png" width="800">

We can see that the algorithm reaches maximum reward of 0 (feature of this environment) in about 200 iterations. Also we need to mention that it wotks similar to Q-Learning algorithm, implementation of which you can see also in this repository. But the main differance between these two is that Q-Learning needs only 4 parameters on each iteration (do not need $A_{t+1}$ - next action), while SARSA need 5 (similar to it's name). 
