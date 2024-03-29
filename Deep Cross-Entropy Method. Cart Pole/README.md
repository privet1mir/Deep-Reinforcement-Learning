# DEEP CROSS-ENTROPY METHOD

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Cart%20Pole/main.ipynb) I have taught agent to solve the [Cart Pole problem](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) from gym by implementing the Cross-Entropy method with deep approach. 

## Description 

This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in [“Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem”](https://ieeexplore.ieee.org/document/6313077). A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

![Cart Pole](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Cart%20Pole/pictures/cart_pole.png)

## Action Space

The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.

| Num | Action  |
| ------- | --- |
| 0 | Push cart to the left | 
| 1 | Push cart to the right | 

Note: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

## Observation Space

The observation is a ndarray with shape (4,) with the values corresponding to the following positions and velocities:

| Num | Observation  | Min | Max |
| ------- | --- | --- | --- |
| 0 | Cart Position | -4.8 | 4.8 |
| 1 | Cart Velocity | -Inf | Inf |
| 2 | Pole Angle | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3 | Pole Angular Velocity | -Inf | Inf |

* The cart x-position (index 0) can be take values between (-4.8, 4.8), but the episode terminates if the cart leaves the (-2.4, 2.4) range.

* The pole angle can be observed between (-.418, .418) radians (or ±24°), but the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)

## Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of +1 for every step taken, including the termination step, is allotted. The threshold for rewards is 500. 

## Starting State

All observations are assigned a uniformly random value in (-0.05, 0.05)

## Episode End

The episode ends if any one of the following occurs:

* Termination: Pole Angle is greater than ±12°

* Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)

* Truncation: Episode length is greater than 500

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

So on the first stage the First Linear Layer takes an input of dimension of states (4 in our case) and gives the 128 outputs. Then ReLU + 1 additional Linear Layer. 

Also, as you can see in the theory above we are interested in policy - matrix of actions probabilities for each state. So we use Softmax function to find out prob-s: 

  ```
        logits = self.forward(state)
        probs = self.softmax(logits).data.numpy()
        action = np.random.choice(self.action_n, p=probs)
  ```

Also loss function is Cross Entropy Loss, as mentioned in the formula above.

## Hyperparameters 

Hyperparameters I chose in this way: 

* the number of trajectories (we don't need many trajectories because we're dealing with the deep approach):

  ```
  trajectory_n = 20
  ```
* the quantile q that helps to choose 'elite' trajectories (we don't have much "right" trajectories that earns maximum reward. It means that we don't loss any information if we set q equal to big number):

  ```
  q_param = 0.9
  ```
* number of iterations (the model needs about 50 iterations to learn the game, so you can set it <. By the way you can play with the number of iterations and number of trajectories):

  ```
  iteration_n = 100
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

![Lerning plot](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Cart%20Pole/pictures/Cart_Pole_graph.png)

We can see that it needs about 50 iterations to reach the maximum mean reward. 

The final result with the total reward of 500 (Which means that the model greatly solve the problem) you can see as a gif: 

![Cart Pole gif](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Deep%20Cross-Entropy%20Method.%20Cart%20Pole/pictures/Cart_Pole.gif)
