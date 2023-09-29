# CROSS-ENTROPY METHOD

In this project we will teach agent to find the route in the labyrinth. I have been use the [gym_maze](https://github.com/MattChanTK/gym-maze/tree/master#gym-maze) library for creating the 2D environment. In this case
I used the 5x5 playing area.


## Environment 

![Environment](https://github.com/privet1mir/Deep-Reinforcement-Learning/blob/main/Cross-Entropy%20Method/lab5x5.png)

$\textbf{Goal: To help agent (who starts from the top left corner) to find its way to the bottom right corner}$



## Theory under the task
For our Environment we have: 

* number of possible states = 25 (5x5 area) - $\textbf{finite state space}$
* number of possible actions = 4 ($\leftarrow \uparrow \downarrow \rightarrow$) - $\textbf{finite action space}$
\
\
The main idea is that we solve the $\textbf{OPTIMIZATION PROBLEM:}$

On each iteration of the algorithm we define the policy in a stohastic way:

$\pi(a|s) = \Pi_{a, s} $, $a \in A, s\in S$, where $S = \\{1, 2, ..., m\\}$ and $A = \\{1, 2, ..., n\\}$

$\Pi_{a, s}$ - Matix m x n which has m rows with possible states (in our case m=25) and n columns with possible actions (n=4). 

The elements of the matrix represent the probality of agent to move in the next state from the current. 

Remember that we working with the $\underline{Markov \ Decision  \ Process \ (MDP)}$: 

$\mathcal{P} (s'|s, a)  = \mathbb{P} [S_{t+1} = s'|S_t = s, A_t = a]$
\
\
As a result we try to solve $\textbf{Finite-dimensional optimization problem}$: 

$\underset{\Pi}{max}f(\Pi)$, where $f(\Pi) = \mathbb{E}_{\pi}[G]$ - the expected value of reward (sum and average over different trajectories for the same policy) 

$\sim$ sampling: $\mathbb{E}_{\pi}[G] \approx \sum_k{G(\tau_k)}$

## How to solve the optimization problem? 

On each iteration we do the following: 

* Policy evaluation. Observing $\mathbb{E}_{\pi}[G]$
* Improve the policy to maximize wealth: $\pi ' \geq \pi $: $\mathbb{E}_{\pi '}[G] \geq \mathbb{E} _ {\pi}[G] $

## How to improve policy?

To improve the policy we select "elite" trajectories. Elite means that their reward is better than some $\gamma_q$ - quantile of the numbers $G(\tau_{k})$. 

So the elite trajectories are: 

$T_{n} = \\{\tau_k, k \in \overline{1, K}: G(\tau_k) > \gamma_k\\}$

if $T_{n} \neq \emptyset$ new policy is: 

$\pi_{n+1}(a|s) = \frac{number \ of \ pairs \ (a|s) \ in \ trajectories \ from \ T_{n}}{number \ of \ s \ in \ trajectories \ from \ T_{n}}$ 

