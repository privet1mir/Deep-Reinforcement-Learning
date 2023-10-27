# DEEP CROSS-ENTROPY METHOD

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Cart%20Pole/main.ipynb) I have taught agent to solve the [Lunar Lander problem](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) from gym by implementing the Cross-Entropy method with deep approach. 

## Description 

This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous (Discrete as default, I also used this one). The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

| Import | gym.make("LunarLander-v2")  |
| ------- | --- |
| Action Space | Discrete(4) | 
| Observation Shape | (8,) | 
| Observation High | [1.5 1.5 5. 5. 3.14 5. 1. 1. ] | 
| Observation Low | [-1.5 -1.5 -5. -5. -3.14 -5. -0. -0. ] | 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Lunar%20Lander/images/lunar_area.png" width="600">

## Action Space

There are four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

## Observation Space

The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Rewards

Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points. If the lander moves away from the landing pad, it loses reward. If the lander crashes, it receives an additional -100 points. If it comes to rest, it receives an additional +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

## Starting State

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

## Episode Termination

The episode finishes if:

1. the lander crashes (the lander body gets in contact with the moon);

2. the lander gets outside of the viewport (x coordinate is greater than 1);

3. the lander is not awake (the body is not awake if it doesn’t move and doesn’t collide with any other body)

## Some theory 

In this project I implement Cross-Entropy method with deep approach. The idea of this methoda and the underlying theory you can find out in my [repository](https://github.com/privet1mir/Deep-Reinforcement-Learning/tree/main/Cross-Entropy%20Method) for the previous (classic) approach. 

The main idea of deep approach is that now the probabilities for the next actions (i.e. policy matrix) predicts by neural network, not by just random search. 

Let $\large F^{\theta}$: $\large \mathbb{R}^n \mapsto \mathbb{R}^m$ be a neural network. We can define our policy as: 

$$\large \pi^{\theta}(i|s) = Softmax(F^{\theta}(s))_i$$ 

Let $\large \theta_0$ be initial parameters, $\large N$ be a number of iterations, $\large K$ be a number of trajectories, $\large q \in (0, 1)$ be a parameter for defning "elite" trajectories, $\large \eta > 0$ be a learning rate.

Then for each $\large n \in \overline{1, N}$ do: 

* (Policy evaluation) Acting in accordance with the policy $\large \pi^{\theta}(i|s)$ get $\large K$ trajectories $\large \tau_k$ and total rewards $\large G(\tau_k)$ . Evaluate $\large \pi_n$:

$$\large \mathbb{E}_{\pi}[G] \approx \frac{1}{K} \sum_k{G(\tau_k)} $$

* (Policy improvement) Select "elite" trajectories:  $\large T_{n} = \\{\tau_k, k \in \overline{1, K}: G(\tau_k) > \gamma_k\\}$ where $\large \gamma_k$ is a $\large q$-quantile of the
numbers $\large G(\tau_k)$, $\large k \in \overline{1, K}$. Define Loss and update parameters by the (stochastic) gradient descent:

$$\large Loss(\theta) = - \frac{1}{|T_n|} \sum_{(a|s) \in T_n}{\ln{\pi^{\theta} (a|s)}}$$


$$\large \theta_{n+1} = \theta{n} - \eta \nabla_{\theta} Loss(\theta_n)$$

## Neural network 

To solve the above problem I used the neural network with the simple structure: 

  ```
self.network = nn.Sequential(nn.Linear(self.state_dim, 128),
                             nn.ReLU(),
                             nn.Linear(128, self.action_n)
                             ) 
  ```

So on the first stage the First Linear Layer takes an input of dimension of states (8 in our case) and gives the 128 outputs. Then ReLU + Linear Layer. It is enough for our agent to learn the environment.

Also, as you can see in the theory above we are interested in policy - matrix of actions probabilities for each state. So we use Softmax function to find out prob-s: 

  ```
        logits = self.forward(state)
        probs = self.softmax(logits).data.numpy()
        action = np.random.choice(self.action_n, p=probs)
  ```

Also loss function is Cross Entropy Loss, as mentioned in the formula above.

## Hyperparameters 

Hyperparameters I chose in this way: 

* the number of trajectories (we have large area, so we need this parameter to be large, you even can try > 100):

  ```
  trajectory_n = 100
  ```
* the quantile q that helps to choose 'elite' trajectories (we don't have much "right" trajectories that earns maximum reward. It means that we don't loss any information if we set q equal to big number):

  ```
  q_param = 0.9
  ```
* number of iterations (the model needs about 160 iterations to learn the game, so you can set it <. By the way you can play with the number of iterations and number of trajectories:

  ```
  iteration_n = 200
  ```

* number of neurons for linear layers (that's ok for our task):

  ```
  neurons_n = 128
  ```
  
* optimizer (I use Adam because of its faster convergence, SGD shows the same +- result):

  ```
  self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
  
  ```


## Results 

![Lerning plot](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Lunar%20Lander/images/reward_lunar.png)

We can see that it needs about 160 iterations to reach the maximum mean reward. 

The output of the model with the total reward of 270 you can see below. 

![Total reward](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Lunar%20Lander/images/total_reward_lunar.png)

The final result with the total reward of 270 (Which means that the model greatly solve the problem) you can see as a gif: 

![Lunar Lander gif](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Lunar%20Lander/images/lunar_lander.gif)

