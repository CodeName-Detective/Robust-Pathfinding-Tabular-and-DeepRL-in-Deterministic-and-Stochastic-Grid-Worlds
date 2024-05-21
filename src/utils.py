import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_pickle(q_table, object_name):
    with open(object_name, "wb") as f:  # Use "wb" for writing binary data
        pickle.dump(q_table, f)
    print('pickle file saved!!!!')

def load_pickle(object_name):
    with open(object_name, "rb") as f:
        deserialized_dict = pickle.load(f)
    return deserialized_dict

def verification_loop(env, agent) -> None:
  """Learning loop to check if Agent can reach GOAL state in the environment after 
  training.

  Args:
      env (GridEnvironment): object of Grid Environment.
      agent (SarsaEpsilonGreedyAgent): object of SARSA Epsilone Greedy Agent that is already trained.
  """
  obs, _ = env.reset()
  terminated, truncated = False, False
  
  print("Initial Observation: {0}".format(obs))
  
  env.render()
  
  # Iterate until the episode ends.
  while not terminated:
    # Taking action as per the optimal policy the algoritham converged to. Hence Epsilon is set to 0.
    action = agent.step(np.argmax(obs),0)
    
    if action == 0:
      action_took = 'Down'
    elif action == 1:
      action_took = 'Up'
    elif action == 2:
      action_took = 'Right'
    elif action == 3:
      action_took = 'Left'
    
    print("Agent opt to take the following action: {0}".format(action_took))
    obs, reward, terminated, truncated, info = env.step(action)
    print("New Observation: {0}; Immediate Reward: {1}, Termination Status: {2}, Termination Message: {3}".format(obs, reward, 
                                                                                                                terminated, info['Termination Message']))

    env.render()
    
    time.sleep(1)
    print('**************')
    

def multi_episodes_verification_loop(env, agent, episodes: int) -> None:
  """Learning loop to check if Agent can reach GOAL state in the environment after 
  training for multiple episodes.

  Args:
      env (GridEnvironment): object of Grid Environment.
      agent (SarsaEpsilonGreedyAgent): object of SARSA Epsilone Greedy Agent that is already trained.
      episodes (int): Number of episodes we should run to verify the agent.
  """
  
  # Iterate over multiple episodes.
  for episode in range(episodes):
    reward_per_episode = 0
    obs, _ = env.reset()
    terminated, _ = False, False
    
    # Iterate untill the episode terminates.
    while not terminated:
      action = agent.step(np.argmax(obs),0)
      obs, reward, terminated, _, _ = env.step(action)
      reward_per_episode += reward
    
    print('For episode {0}, the cummulative reward is: {1}'.format(episode+1, reward_per_episode))


def main_plot(reward_across_episodes: list)-> None:
  """Plots the graph of reward across episodes.

  Args:
      reward_across_episodes (list): List of cummulative reward across episodes.
  """
  plt.figure(figsize=(10, 7))
  plt.plot(reward_across_episodes, 'ro')
  plt.xlabel('Episode')
  plt.ylabel('Reward Value')
  plt.title('Rewards Per Episode (Training)')
  plt.grid()
  plt.show()


def extra_plots(reward_across_episodes: list, epsilons_across_episodes: list) -> None:
  """Plot cummulative rewards across epsiodes and epsilon decay across episodes. 

  Args:
      reward_across_episodes (list): List of cummulative reward across episodes.
      epsilons_across_episodes (list): List of epsilons used across episodes.
  """
  plt.figure(figsize=(15, 5))

  plt.subplot(1,2,1)
  plt.plot(reward_across_episodes)
  plt.xlabel('Episode')
  plt.ylabel('Cummulative Reward Per Episode')
  plt.title('Cummulative Reward vs Episode')
  plt.grid()

  plt.subplot(1,2,2)
  plt.plot(epsilons_across_episodes)
  plt.xlabel('Episode')
  plt.ylabel('Epsilon Values')
  plt.title('Epsilon Decay')
  plt.grid()


def run_learned_policy(env, agent, double_q_learning=False):
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    
    print("Initial state: {0};".format(obs.reshape((4, 5))))
    
    total_reward = 0
    steps = 0
    
    # Continue through the episode until we reach a termination state
    while not terminated:
        # Agent decides on what action to choose
        if double_q_learning:
            action = np.argmax((agent.q_table_a[np.argmax(obs),:]+agent.q_table_b[np.argmax(obs),:])/2)
        else:
            action = np.argmax(agent.q_table[np.argmax(obs),:])
        
        # Mapping action number to action
        action_names = ['Down', 'Up', 'Right', 'Left']
        action_took = action_names[action]
        print("Agent opts to take the following action: {0}".format(action_took))
        
        # Environment performs taking an action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        print("New Observation: {0}; Immediate Reward: {1}, Termination Status: {2}, Termination Message: {3}".format(obs.reshape((4, 5)), reward, 
                                                                                                                        terminated, info['Termination Message']))
        time.sleep(1)
        print('**************')
        steps += 1
    print("Total Reward Collected Over the Episode: {0} in Steps: {1}".format(total_reward, steps))

def run_learned_policy_supressed_printing(env, agent, double_q_learning=False):
    obs, _ = env.reset()
    terminated, truncated = False, False

    
    total_reward = 0
    steps = 0
    
    # Continue through the episode until we reach a termination state
    while not terminated:
        # Agent decides on what action to choose
        if double_q_learning:
            action = np.argmax((agent.q_table_a[np.argmax(obs),:]+agent.q_table_b[np.argmax(obs),:])/2)
        else:
            action = np.argmax(agent.q_table[np.argmax(obs),:])
        
        # Mapping action number to action
        action_names = ['Down', 'Up', 'Right', 'Left']
        action_took = action_names[action]
        
        # Environment performs taking an action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
    return total_reward



def plot_grid(env, agent, reward_across_episodes: list, epsilons_across_episodes: list, double_q_learning: bool = False) -> None:
    """Plot main and extra plots in a 4x4 grid."""
    
    
    total_reward_learned_policy = []
    for i in range(30):
        if double_q_learning:
            total_reward_learned_policy.append(run_learned_policy_supressed_printing(env,agent, double_q_learning))
        else:
            total_reward_learned_policy.append(run_learned_policy_supressed_printing(env,agent))
        
    plt.figure(figsize=(15, 10))

    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(reward_across_episodes, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward Value')
    plt.title('Rewards Per Episode (Training)')
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(total_reward_learned_policy, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Reward Value')
    plt.title('Rewards Per Episode (Learned Policy Evaluation)')
    plt.grid()

    # Extra plots
    plt.subplot(2, 2, 3)
    plt.plot(reward_across_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Reward Per Episode (Training)')
    plt.title('Cummulative Reward vs Episode')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(epsilons_across_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Values')
    plt.title('Epsilon Decay')
    plt.grid()

    plt.tight_layout()
    plt.show()


def run_env(env, online_value_function, episodes = 100):
    
    reached_end_goal = 0
    rewards = []
    
    for episode in range(episodes):
        observation, _ = env.reset()
        terminated = False
        
        reward_per_episode = 0
        
        # For each episode
        while not terminated:
            # Make a Transistion
            action = torch.argmax(online_value_function.act(observation), dim=1).item()
            new_observation, reward, terminated, truncated, info = env.step(action)
            reward_per_episode += reward
            observation = new_observation
        
        if info['Termination Message'] == 'Goal Position Reached !!!':
            reached_end_goal += 1
        rewards.append(reward_per_episode)
    
    env.close()
    return rewards