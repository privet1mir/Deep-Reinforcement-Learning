# Monte-Carlo Algorithm

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Monte-Carlo%20Algorithm.%20Taxi-v3/Monte_Carlo_algorithm.ipynb) I introduce monte-carlo algorithm that can be used for solving deterministic environments, such as gym taxi, that we will explore this time. 

## Theory
Let's $Q(s,a) = 0$, $N(s,a) = 0$ and $\varepsilon = 1$.

For each episode $k \in \overline{1,K}$ we do:

1. Because of $\pi = \varepsilon\text{-greedy}(Q)$ we get trajectories $\tau = (S_0,A_0,\ldots,S_T)$ and rewards $(R_0,\ldots,R_{T-1})$. Then we can determine $(G_0,\ldots,G_{T-1}):$
   
$$
G_t = \sum\limits_{k=t}^{T-1} \gamma^{k-t} R_t,\quad G_{T-1} = R_{T-1},\quad G_{T-2} = R_{T-2} + \gamma R_{T-1},\quad G_i = R_i + \gamma G_{i+1},\quad G_{T} = Q(S_T,\pi_{greedy}(S_T)).
$$

3. For each $t \in \overline{0,T-1}$ we improve $Q$ и $N$:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac{1}{N(S_t,A_t) + 1}\big(G_t - Q(S_t,A_t)\big),
$$

$$
N(S_t,A_t) \leftarrow N(S_t,A_t) + 1
$$

Reduce $\varepsilon$ by law that we want (it's sometimes better to use soft decreasing law in epsilon, f.e. tanh or sigmoid with boundaries between 1 and 0). 


## Practice. Solve Taxi-v3

In this project I've implemented the Monte-Carlo algorithm to solve the [Taxi-v3](https://www.gymlibrary.dev/environments/toy_text/taxi/) environment from gym.

<img src="https://www.gymlibrary.dev/_images/taxi.gif" width="500">

## Results

We can see that Monte-Carlo algorithm is not an optimal to solve this deterministic area. It often needs a lot of iterations and parameters tunning to solve such environments. For taxi case the best result that I reached you can see below: 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Monte-Carlo%20Algorithm.%20Taxi-v3/images/MC_reward.png" width="800">

Also we can compare it with two other algorithms, SARSA and Q-Learning that can solve this task. 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Monte-Carlo%20Algorithm.%20Taxi-v3/images/comparison_graph1.png" width="800">

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Monte-Carlo%20Algorithm.%20Taxi-v3/images/comparison_graph2.png" width="800">

The realisation of SARSA and Q-Learning algorithms you can also see in this course repository. 
