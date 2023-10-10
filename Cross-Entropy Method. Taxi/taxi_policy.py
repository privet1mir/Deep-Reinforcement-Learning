import gym
import gym_maze
import numpy as np
import random
import time 
import matplotlib.pyplot as plt
import seaborn as sns


env = gym.make('Taxi-v3')

state_n = 500 
action_n = 6 



class CrossEntropyAgent():
    #agent's moves fully defines by policy matrix [MxN where M - state, N - actions]
    def __init__(self, state_n, action_n, lambdaa=0.5):
        self.lambdaa = lambdaa
        self.state_n = state_n
        self.action_n = action_n
        #initial policy
        self.model = np.ones((state_n, action_n)) / action_n
        
    def get_action(self, state):
        #sampling
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)
    
    #policy changing
    def fit(self, elite_trajectories): 
        new_model = np.zeros((self.state_n, self.action_n)) #array thay we'll fill through elite traj-s
        for trajectory in elite_trajectories: 
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1
                
        #normalize sum(each row) == 1
        
       
        for state in range(self.state_n): 
            if np.sum(new_model[state]) > 0: 
                new_model[state] /= np.sum(new_model[state])
            else: 
                new_model[state] = self.model[state].copy()
                
        new_model = (self.lambdaa * new_model.copy()) + ((1-self.lambdaa) * self.model)
        
        self.model = new_model
        
        return None
        
        

#let's get the trajectory
def get_trajectory(env, agent, max_len=1000, visualize=False): 
    trajectory = {'states':[], 'actions': [], 'rewards': []}
    
    obs = env.reset()
    state = obs
    
    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        
        state = obs
        
        if visualize: 
            time.sleep(0.2)
            env.render()
        
        if done: 
            break
        
    return trajectory
    
agent = CrossEntropyAgent(state_n, action_n)
q_param = 0.8
trajectory_n = 1000
iteration_n = 20

mean_rewards = [] 

for iteration in range(iteration_n): 
    
    #policy evaluation
    
    #the array of trajectories
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    #the array of total reward 
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    #and we need average
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))
    
    mean_rewards.append(np.mean(total_rewards))
    
    #policy improvement (gamma-q)
    quantile = np.quantile(total_rewards, q_param)
    #remain only elite trajectories
    elite_trajectories = []
    for trajectory in trajectories: 
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile: 
            elite_trajectories.append(trajectory)
            
            
    agent.fit(elite_trajectories)
          
        
        
    
trajectory = get_trajectory(env, agent, max_len=500, visualize=False)
print('total reward:', sum(trajectory['rewards']))
print('model:')
print(agent.model)

# Create plot
plt.plot(mean_rewards, color='green', marker='o', linestyle='--')

# Add title and labels
plt.title('Learning plot')
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.xticks(range(0, iteration_n+1, 2))

# Add grid
plt.grid(True)
plt.text(2, 8, 'Lambda = 0.5')
plt.savefig('Taxi_policy_smoothing.png')