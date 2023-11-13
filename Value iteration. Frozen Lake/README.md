# VALUE ITERATION METHOD

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Value%20iteration.%20Frozen%20Lake/Value_Iteration.ipynb) I have to implement the policy iteration algorithm to solve the [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) gym environment.

<img src="https://www.gymlibrary.dev/_images/frozen_lake.gif" width="350">

The environment for this project you can obtatain at [Frozen_Lake.py](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Value%20iteration.%20Frozen%20Lake/Frozen_Lake.py). We will not use the directly gym environment because of some problems with reward in it. Luckily, we have working one.

In this project I implement Value Iteration Algorithm, while in the previous project I did [Policy Iteration](https://github.com/privet1mir/Deep-Reinforcement-Learning/tree/main/Policy%20iteration.%20Frozen%20Lake) one.

## Theory. Value Iteration algorithm.

The whole underlying theory of value and action functions you can in my [Policy Iteration](https://github.com/privet1mir/Deep-Reinforcement-Learning/tree/main/Policy%20iteration.%20Frozen%20Lake) project. However, this time I implement value iteration algorithm, so we need to discuss the main idea.

The solution can be devided into two parts. 

### 1 Part - Value iteration:

Let $\large{v^0(s), s\in S}$ and $\large{K \in \mathbb{N}}$.

For each $\large{k \in \overline{0, K},}$ do:

$$\large{v^{k+1}(s) = \underset{a \in A}{max} (R(s, a) + \gamma \sum_{s'} P(s'|s, a) v^k(s'))}$$

If reward depends on s': (our case)

$$\large{v^{k+1}(s)  = \underset{a \in A}{max} \sum_{s'} P(s'|s, a) \Big( R(s, a, s') + \gamma  v^k(s')\Big)}$$

Then we have theorem:

$$\large{v^k \rightarrow v_*, k \rightarrow \infty \text{: convergence rate } O(mn^2)}$$

Then we can obtain q-values as in policy iteration case:

$$\large{v_*(s) = \underset{a \in A}{max} q_{\*} (s,a)}$$


$$\large{q_*(s, a) = R(s, a) + \gamma \sum_{s'}P(s'|s, a) v_{\*}(s')}$$


### 2 Part - Greedy Policy Improvement:

$$\large{\pi_*(a|s) = \begin{cases}
  1, & \text{if } a\in argmax_{a' \in A} q_{\*} (s, a') \\    
  0, & \text{otherwise}   
\end{cases}}$$

## Frozen Lake

### Description

Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H) by walking over the Frozen(F) lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.

### Action Space

The agent takes a 1-element vector for actions. The action space is (dir), where dir decides direction to move in which can be:

* left
* down
* right
* up

### Observation Space
 
For the 4x4 map we have 16 possible observations. Which can be described by 2-dimensions (x-coordinate, y-coordinate). 

### Rewards

Reward schedule:

* Reach goal(G): +1

* Reach hole(H): 0

* Reach frozen(F): 0

### Results 

After value iteration alhorithm realisation we can look at mean total reward graph for different $\gamma \in (0, 1)$ - discount factor. As a parameters I set: 

  ```
iter_n = 100
eval_iter_n = 100
  ```

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Value%20iteration.%20Frozen%20Lake/images/value_iteration_graph_1.png" width="500">


<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Value%20iteration.%20Frozen%20Lake/images/value_iteration_graph_2.png" width="500">


As we can see there is a good improvement in mean total reward if we compare it with the policy iteration result. However we need to understand that it is not a good idea to compare these algorithms just by setting parameters equal. We have different number of times interacting with the area for these algorithms. We can compare these two methods by checking total rewards for equal interacting times. I implement this by writing a counter in a get_q_values function. This function is used for the environment interaction. So, by choosing parameters in right way (different iteration and eval iteration numbers for each of the algorithm), we can compare them: 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Value%20iteration.%20Frozen%20Lake/images/policy%20counter.png" width="500">

  
<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Value%20iteration.%20Frozen%20Lake/images/value%20counter.png" width="500">

We can see that they have the same +- mean total reward. However value iteration algorithm has better precision if we play with the parameters in my case. Also for the final envestigations we need to compare these algorithms for a variety set of interactions. However, maybe it will be done in future. 
