# CROSS-ENTROPY METHOD

In this project I have taught agent to solve the [taxi problem](https://www.gymlibrary.dev/environments/toy_text/taxi/) from gym. 

## Description 

There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). 
When the episode starts, the taxi starts off at a random square and the passenger is at a random location. 
The taxi drives to the passenger’s location, picks up the passenger, drives to the passenger’s destination (another one of the four specified locations), 
and then drops off the passenger. Once the passenger is dropped off, the episode ends.


![Taxi environment](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/taxi_area.png)


## Actions and states 

There are 6 discrete deterministic actions:

* move south

* move north

* move east

* move west

* pickup passenger

* drop off passenger

There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.

## Rewards 

* -1 per step unless other reward is triggered.

* +20 delivering passenger.

* -10 executing “pickup” and “drop-off” actions illegally.
  
## Hyperparameters

The theory lying under the algorithm you can find in my [previous project](https://github.com/privet1mir/Deep-Reinforcement-Learning/tree/main/Cross-Entropy%20Method) connected with the cross-entropy method. 

In this task I choose the hyperparameters of the model in such way: 

* the number of trajectories (we have large area, if we pick less trajectories - the model won't be able to learn the right path):

  ```
  trajectory_n = 1000
  ```
* the quantile q that helps to choose 'elite' trajectories (we don't have much "right" trajectories that earns maximum reward. It means that we don't loss any information if we set q equal to big number):

  ```
  q_param = 0.8
  ```
* number of iterations (the model needs only ~7 iterations to learn the area, so you can set it <. By the way you can play with the number of iterations and number of trajectories):

  ```
  iteration_n = 30
  ```

## Results 

![Lerning plot](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/taxi_graph.png)

We can see that it needs only 7 iterations (with such big number of trajectories) to reach the maximum reward. 

The final result with the total reward of 6 I can show you as a gif:)

![Taxi gif](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/taxi.gif)


## How to improve? - Regularization 

One of the weaknesses of the algorithm is that $\textbf{the policy update is highly dependent on randomness}$

It means that there is a chance that the algorithm cannot learn the right moves because there exist 0-probability actions after the policy updating. And it will continue for all following policy updates. 

Such case you can see on the gif below (the agent wrongly learn this step because the probability of right move is equel to 0, which connects with the wrong policy updating)

![Wrong move](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/taxi_wrong_move.gif)



The main of idea of the regularization is to add some stochasticity for each policy. In this way we may have the less reward and need some more time for algorithm convergence, but there is no chance to lose different actions in determent states, because the probability:

$\mathcal{p}(a|s) \neq 0$ for $\forall {a} \in A, \forall {s} \in S$


### Laplace smoothing

$$\Huge \pi_{n+1} (a|s) = \frac{|(a|s) \in {T_n}| + \lambda}{|s \in T_n| + \lambda |A|}, \lambda >0$$

You can see this [implementation](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/taxi_laplace.py) with the $\lambda = 0.5$ (It means moderate smoothing. This makes probabilities smoother and prevents null probabilities, while allowing empirical evidence to significantly influence policy) we can see that our final policy has no zero probabilities - what we want.

![model](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/laplace_model.png)

And the training process of our model: 

![model train laplace](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/laplace_smoothing_graph.png)



### Policy smoothing

$$\Huge \pi_{n+1} (a|s) = \lambda \pi_{n+1} (a|s) + (1-\lambda)\pi_{n} (a|s) , \lambda \in (0, 1]$$


![model train policy](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method.%20Taxi/policy_smoothing_graph.png)
