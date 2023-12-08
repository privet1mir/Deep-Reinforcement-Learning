# Deep Q-Learning Algorithm

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/DQN_algo.ipynb) I introduce Deep Q-Learning algorithm (DQN) that can be used for solving dofferent environments, such as Lunar Lander, that we will explore this time. 
Also I implement some [improvements](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/Modernizations_of_DQN.ipynb) of these algorithm (including Double DQN that show's the best accuracy). So the structure of these project will be the following: 

* Theory behind Deep Q Learning Algorithm
* DQN with Hard Target Update
* DQN with Soft Target Update
* Double DQN
* Description of the area
* Results

## Theory

### DQN

We implement the approximation structure $Q^\theta$, initial parameters vector $\theta$, probability of environment exploration $\varepsilon = 1$.

For each episode $k$ do:

While episode not done:

- Being in state $S_t$ we do action $A_t \sim \pi(\cdot|S_t)$, where $\pi = \varepsilon\text{-greedy}(Q^\theta)$, receive reward $R_t$  move to state $S_{t+1}$. Save $(S_t,A_t,R_t,S_{t+1}) \rightarrow Memory$


- Take $\{(s_i,a_i,r_i,s'_i)\}\_{i=1}^{n} \leftarrow Memory$, obtain targets:

$$\large{y = \begin{cases}
  r_i,& \text{ if } s_i ' \text{ is a terminal state} \\    
  r_i + \gamma \underset{a'}{\max} Q^{\theta}(s_i ', a'),& \text{ otherwise } 
\end{cases}}$$

Loss function $Loss(\theta) = \frac{1}{n}\sum\limits_{i=1}^n \big(y_i - Q^\theta(s_i,a_i)\big)^2$
and upgrade the parameters vectors

$$
\theta \leftarrow \theta - \alpha \nabla_\theta Loss(\theta)
$$

- Decrease $\varepsilon$

### DQN with Hard Target Update

* We set some parameter $\theta = \theta '$ (means parameters of Neural Network, it can be done by using state_dict)

* Do a lot of iterations:

    * $y = r + \gamma \underset{a'}{max} Q^{\theta'} (s', a')$
    * $Loss(\theta) = (y - Q^{\theta} (s, a))^2$
    * $\theta \leftarrow \theta - \alpha \nabla_{\theta} Loss(\theta) $

* Update $\theta' = \theta$ (it can be done by using load_state_dict)

### DQN with Soft Target Update

* $y = r + \gamma \underset{a'}{max} Q^{\theta'} (s', a')$

* $Loss(\theta) = (y - Q^{\theta} (s, a))^2$

* $\theta \leftarrow \theta - \alpha \nabla_{\theta} Loss(\theta) $

* $\theta ' = \tau \theta + (1-\tau) \theta'$

### Double DQN

* $y = r + \gamma Q ^ {\theta} (s', \underset{a'}{argmax} Q^{\theta'} (s', a'))$

* $Loss(\theta) = (y - Q^{\theta} (s, a))^2$

* $\theta \leftarrow \theta - \alpha \nabla_{\theta} Loss(\theta)$

* $\theta ' = \tau \theta + (1-\tau) \theta'$

## Description of the area

This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous (Discrete as default, I also used this one). The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

| Import | gym.make("LunarLander-v2")  |
| ------- | --- |
| Action Space | Discrete(4) | 
| Observation Shape | (8,) | 
| Observation High | [1.5 1.5 5. 5. 3.14 5. 1. 1. ] | 
| Observation Low | [-1.5 -1.5 -5. -5. -3.14 -5. -0. -0. ] | 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Lunar%20Lander/images/lunar_area.png" width="600">

### Action Space

There are four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

### Observation Space

The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

### Rewards

Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points. If the lander moves away from the landing pad, it loses reward. If the lander crashes, it receives an additional -100 points. If it comes to rest, it receives an additional +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

### Starting State

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

### Episode Termination

The episode finishes if:

1. the lander crashes (the lander body gets in contact with the moon);

2. the lander gets outside of the viewport (x coordinate is greater than 1);

3. the lander is not awake (the body is not awake if it doesn’t move and doesn’t collide with any other body)

## Results

### DQN 

For Deep Q-Learning algorithm I used neural network with the following structure: 

```
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()
```
Such network is enough to solve environments like Lunar Lander from gym. Hyperparameters I set in this way: 

| Hyperparameter | Value |
| ------- | --- |
| gamma | 0.99 | 
| lr | $1e^{-3}$ | 
| batch size | 64 | 
| epsilon decrease | 0.01 | 
| epsilon min | 0.01 |
| episode_n | 500 |
| t_max | 500 |

After implemeting DQN alhorithm we can see how it works by plotting reward/iteration graph: 

![DQN reward graph](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/images/DQN%20reward%20plot.png)

We can see that DQN solves the area in about 300 iterations (means reaching the maximum reward ~250). However it is not stable, we can see strong fluctuations in the tail of the reward graph. We can try to fix it using some improvements you can see below. 

### DQN with Hard Target Update
![DQN with Hard Target Update](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/images/Hard%20Target%20reward%20plot.png)



###  DQN with Soft Target Update
![DQN with Soft Target Update](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/images/Soft%20Target%20reward%20plot.png)

### Double DQN

![Double DQN](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/images/Double%20DQN%20reward%20plot.png)

We can see that Hard target update algorithm and Double DQN strongly improved the perfomance of the DQN algorithm. The solution we know get is more stable than we have in "clear" DQN. And, what is more important, implementer algorithm works really fast, for example, in comparison with the deep cross-entropy method, realisation of which you can also find in my [repository](https://github.com/privet1mir/Deep-Reinforcement-Learning/tree/main/Deep%20Cross-Entropy%20Method.%20Lunar%20Lander). 

![Learning process](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Q%20Learning%20(DQN).%20Double%20DQN/images/Double%20DQN%20Lunar%20Lander.gif)
