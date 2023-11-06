# DEEP CROSS-ENTROPY METHOD

In this [project](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Policy%20iteration.%20Frozen%20Lake/Policy_iteration.ipynb) I have to implement the policy iteration algorithm to solve the [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) gym environment.

<img src="https://www.gymlibrary.dev/_images/frozen_lake.gif" width="350">

The environment for this project you can obtatain at [Frozen_Lake.py](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Policy%20iteration.%20Frozen%20Lake/Frozen_Lake.py). We will not use the directly gym environment because of some problems with reward in it. Luckily, we have working one.

The project will be devided into several parts. First off all we need to discus the underlying Policy iteration method theory. Then we will try to implent this in Frozen Lake environment. 

## Theory. Policy Iteration algorithm.

The upper level idea behind the policy iteration algorithm is that we can update our policy by maximized Action-Value Function (q - function), so at the end we will have deterministic optimal policy without any stohasticity. 

First of all let's define value function: $\large v_{\pi} (s) = \mathbb{E}_{\pi} (G)$ - expected value of our reward: $\large G = \sum_t {\gamma ^ t R(S_t, A_t)}$, in this formula: $\large \gamma \in [0, 1]$ - discount coefficient, $\large R(S_t, A_t)$ - reward which we get if we move from state $\large S_t$ with action $\large A_t$. It is clear that for determenistic policy we have just G.

Action value fucntion we can define such as: $\large q_{\pi} (s, a) = \mathbb{E}_{\pi} [G| S_0 = s, A_0 = a]$. And we can connect it with the value fucntion using Bellman Equation: 

$$\large q_{\pi} (s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a)v_{\pi}(s')$$

$$\large v_{\pi}(s) = \sum_{a} \pi (a|s) q_{\pi} (s, a)$$

Now, when we know how to obtain value and action-value functions, we can just follow the steps: 

1. Iterative Policy Evaluation:

$$\large v^{l+1} = R_{\pi^k} + P_{\pi^k} v^l \text{ for } l \in \overline{0, L-1}$$

Then we can define $\large q^L (s, a)$ by $\large v^L (s)$ (as before).

2. Greedy Policy Improvement:

$$\large \pi^{k+1}(a|s) = \begin{cases}
  1, & \text{if } a\in argmax_{a' \in A} q^L (s, a') \\    
  0, & \text{otherwise}   
\end{cases}
$$

Final policy founded by greedy improvement will be optimal policy for our task. 

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

By implementing [this](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Policy%20iteration.%20Frozen%20Lake/Policy_iteration.ipynb) we can see that after 500 iterations of our game with optimal policy we have mean total reward = 0.874. That conects with the fact that since we have deterministic optimal policy, we are working with the stohastic action space, we always has non-zero probability for each action to move not in right direction (we have frozen lake), so it decreases our mean reward. For example: 

  ```
state = (3, 2)
action = 'right'
env.get_next_states(state, action)

#(1, 1) with prob 0.1
#(0, 2) with prob 0.8
#stay in the same with prob 0.1
  ```

However we have mean reward of 0.874, which we can interpet in the way that our agent do about 87% right steps in average. 

Also we can see the final policy that solves our task: 

<img src="https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Policy%20iteration.%20Frozen%20Lake/images/final_policy.png" width="500">

As I mentioned above such probabilities in our policy connected with the stohastic action space. 
  
