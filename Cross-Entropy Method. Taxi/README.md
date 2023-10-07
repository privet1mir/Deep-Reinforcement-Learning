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