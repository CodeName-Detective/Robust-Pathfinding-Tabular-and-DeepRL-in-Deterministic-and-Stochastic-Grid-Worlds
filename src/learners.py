import numpy as np

from environments import GridEnvironment
from src.agents import (
    DoubleQLearningGreedyAgent,
    NStepSarsaGreedyAgent,
    QLearningGreedyAgent,
    SarsaEpsilonGreedyAgent,
)


def sarsa_learning_loop(max_timesteps: int,learning_rate: float, discount_factor: float, episodes: int,
                        min_epsilon_allowed: float, initial_epsilon_value: float) -> tuple['SarsaEpsilonGreedyAgent', list, list]:
  """Learning loop train Agent to reach GOAL state in the environment using SARSA Algorithm.

  Args:
      max_timesteps (int): Maximum timesteps allowed per episode.
      learning_rate (float): Learning rate used in SARSA algorithm
      discount_factor (float): Discount factor to quantify the importance of future reward.
      episodes (int): Number of episodes we should train.
      min_epsilon_allowed (float): Minimum epsilon that we should reach by the end of the training.
      initial_epsilon_value (float): Initial epsilon that we should use while starting the learning. 

  Returns:
      tuple[SarsaEpsilonGreedyAgent, list, list]: Returns a tuple containing agent,
                                                  cumulative rewards across episodes,
                                                  epsilon used across episodes respectively.
  """
  
  # Initiating environment and agent.
  env = GridEnvironment(max_timesteps=max_timesteps)
  agent = SarsaEpsilonGreedyAgent(env, learning_rate=learning_rate, discount_factor = discount_factor)
  
  # Initiating Number of episodes, epsilon values.
  episodes = episodes

  epsilon = initial_epsilon_value
  min_epsilon_allowed = min_epsilon_allowed
  
  # Calculating Epsilon Decay factor. 
  epsilon_decay_factor = np.power(min_epsilon_allowed/epsilon, 1/episodes)
  
  # Initiating list to store rewards and epsilons we use across episodes
  reward_across_episodes = []
  epsilons_across_episodes = []
  
  # Iterating over Episodes.
  for _ in range(episodes):
    # Resetting the environment.
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    # Fectcing Current State and Current Action details.
    current_state = np.argmax(obs)
    current_action = agent.step(current_state, epsilon)
    
    reward_per_episode = 0
    epsilons_across_episodes.append(epsilon)
    
    # Iterating over an epsidoe untill termination status is reached.
    while not terminated:
      # Taking one step in the environment.
      obs, reward, terminated, truncated, _ = env.step(current_action)
      
      # Calculating cummulative reward for an Epoch.
      reward_per_episode = reward_per_episode+reward
      
      # Fetching future state and future reward.
      future_state = np.argmax(obs)
      future_action = agent.step(future_state, epsilon)
      
      # Updating Q values.
      agent.update_qvalue(current_state, current_action, reward, future_state, future_action)

      current_state = future_state

      current_action = future_action
    
    # Decaying Epsilon
    epsilon = epsilon_decay_factor*epsilon
    reward_across_episodes.append(reward_per_episode)

  return agent, reward_across_episodes, epsilons_across_episodes


def q_learning_learning_loop(max_timesteps: int,learning_rate: float, discount_factor: float, episodes: int,
                        min_epsilon_allowed: float, initial_epsilon_value: float) -> tuple['QLearningGreedyAgent', list, list]:
  """Learning loop train Agent to reach GOAL state in the environment using Q-Learning Algorithm.

  Args:
      max_timesteps (int): Maximum timesteps allowed per episode.
      learning_rate (float): Learning rate used in SARSA algorithm
      discount_factor (float): Discount factor to quantify the importance of future reward.
      episodes (int): Number of episodes we should train.
      min_epsilon_allowed (float): Minimum epsilon that we should reach by the end of the training.
      initial_epsilon_value (float): Initial epsilon that we should use while starting the learning. 

  Returns:
      tuple[QLearningGreedyAgent, list, list]: Returns a tuple containing agent,
                                                  cumulative rewards across episodes,
                                                  epsilon used across episodes respectively.
  """
  
  # Initiating environment and agent.
  env = GridEnvironment(max_timesteps=max_timesteps)
  agent = QLearningGreedyAgent(env, learning_rate=learning_rate, discount_factor = discount_factor)
  
  # Initiating Number of episodes, epsilon values.
  episodes = episodes

  epsilon = initial_epsilon_value
  min_epsilon_allowed = min_epsilon_allowed
  
  # Calculating Epsilon Decay factor. 
  epsilon_decay_factor = np.power(min_epsilon_allowed/epsilon, 1/episodes)
  
  # Initiating list to store rewards and epsilons we use across episodes
  reward_across_episodes = []
  epsilons_across_episodes = []
  
  # Iterating over Episodes.
  for _ in range(episodes):
    # Resetting the environment.
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    # Fectcing Current State and Current Action details.
    current_state = np.argmax(obs)
    current_action = agent.step(current_state, epsilon)
    
    reward_per_episode = 0
    epsilons_across_episodes.append(epsilon)
    
    # Iterating over an epsidoe untill termination status is reached.
    while not terminated:
      # Taking one step in the environment.
      obs, reward, terminated, truncated, _ = env.step(current_action)
      
      # Calculating cummulative reward for an Epoch.
      reward_per_episode = reward_per_episode+reward
      
      # Fetching future state and future reward.
      future_state = np.argmax(obs)
      future_action = agent.step(future_state, epsilon)
      
      # Updating Q values.
      agent.update_qvalue(current_state, current_action, reward, future_state)

      current_state = future_state

      current_action = future_action
    
    # Decaying Epsilon
    epsilon = epsilon_decay_factor*epsilon
    reward_across_episodes.append(reward_per_episode)

  return agent, reward_across_episodes, epsilons_across_episodes

def n_step_sarsa_learning_loop(max_timesteps: int,learning_rate: float, discount_factor: float, episodes: int,
                        min_epsilon_allowed: float, initial_epsilon_value: float, n:int = 2) -> tuple['NStepSarsaGreedyAgent', list, list]:
  """Learning loop train Agent to reach GOAL state in the environment using N-step SARSA Algorithm.

  Args:
      max_timesteps (int): Maximum timesteps allowed per episode.
      learning_rate (float): Learning rate used in SARSA algorithm
      discount_factor (float): Discount factor to quantify the importance of future reward.
      episodes (int): Number of episodes we should train.
      min_epsilon_allowed (float): Minimum epsilon that we should reach by the end of the training.
      initial_epsilon_value (float): Initial epsilon that we should use while starting the learning. 
      n(int, optional): n-step sarasa n parameter. Defaults to 2.

  Returns:
      tuple[NStepSarsaGreedyAgent, list, list]: Returns a tuple containing agent,
                                                  cumulative rewards across episodes,
                                                  epsilon used across episodes respectively.
  """
  
  # Initiating environment and agent.
  env = GridEnvironment(max_timesteps=max_timesteps)
  agent = NStepSarsaGreedyAgent(env, learning_rate=learning_rate, discount_factor = discount_factor)
  
  # Initiating Number of episodes, epsilon values.
  episodes = episodes

  epsilon = initial_epsilon_value
  min_epsilon_allowed = min_epsilon_allowed
  
  # Calculating Epsilon Decay factor. 
  epsilon_decay_factor = np.power(min_epsilon_allowed/epsilon, 1/episodes)
  
  # Initiating list to store rewards and epsilons we use across episodes
  reward_across_episodes = []
  epsilons_across_episodes = []
  
  
  # Iterating over Episodes.
  for _ in range(episodes):
    # Resetting the environment.
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    # Fectcing Current State and Current Action details.
    current_state = np.argmax(obs)
    current_action = agent.step(current_state, epsilon)
    
    reward_per_episode = 0
    reward = 0
    epsilons_across_episodes.append(epsilon)
    
    episode_path = [{'reward': 0, 'state': current_state, 'action': current_action}]
    
    
    # Iterating over an epsidoe untill termination status is reached.
    while not terminated:
      # Taking one step in the environment.
      obs, reward, terminated, truncated, _ = env.step(current_action)
      
      # Calculating cummulative reward for an Epoch.
      reward_per_episode = reward_per_episode+reward
      
      # Fetching future state and future reward.
      future_state = np.argmax(obs)
      future_action = agent.step(future_state, epsilon)
      
      episode_path.append({'reward': reward, 'state': future_state, 'action': future_action})

      current_state = future_state

      current_action = future_action
      
    windows = []
    end_of_episode = False  
    
    for window_start in range(len(episode_path) - n + 2):
        if window_start == len(episode_path) - n + 1:  
            end_of_episode = True
        else:
            windows.append(episode_path[window_start:window_start+n-1] + [episode_path[window_start+n-1]])
        
        if end_of_episode:
            for offset_idx in range(5):
                if window_start+offset_idx < len(episode_path)-1:
                    windows.append(episode_path[window_start+offset_idx:])
    
    for window in windows:
      reward_return = 0
      for idx, last_step_of_window in enumerate(window[::-1]):
        reward_return += (np.power(discount_factor,len(window)-idx-1)*last_step_of_window['reward'])
      reward_return += np.power(discount_factor,len(window))*agent.q_table[window[-1]['state'], window[-1]['action']]
      
      # Update Q_value
      agent.update_qvalue(window[0]['state'], window[0]['action'], reward_return)
    
    
    # Decaying Epsilon
    epsilon = epsilon_decay_factor*epsilon
    reward_across_episodes.append(reward_per_episode)

  return agent, reward_across_episodes, epsilons_across_episodes



def q_learning_learning_loop_env(env,learning_rate: float, discount_factor: float, episodes: int,
                        min_epsilon_allowed: float, initial_epsilon_value: float) -> tuple['QLearningGreedyAgent', list, list]:
  """Learning loop train Agent to reach GOAL state in the environment using Q-Learning Algorithm.

  Args:
      env (gymnasium.Env): object of Grid Environment.
      learning_rate (float): Learning rate used in SARSA algorithm
      discount_factor (float): Discount factor to quantify the importance of future reward.
      episodes (int): Number of episodes we should train.
      min_epsilon_allowed (float): Minimum epsilon that we should reach by the end of the training.
      initial_epsilon_value (float): Initial epsilon that we should use while starting the learning. 

  Returns:
      tuple[QLearningGreedyAgent, list, list]: Returns a tuple containing agent,
                                                  cumulative rewards across episodes,
                                                  epsilon used across episodes respectively.
  """
  
  # Agent.
  agent = QLearningGreedyAgent(env, learning_rate=learning_rate, discount_factor = discount_factor)
  print("Initial Q-Table; {0}".format(agent.q_table))
  
  # Initiating Number of episodes, epsilon values.
  episodes = episodes

  epsilon = initial_epsilon_value
  min_epsilon_allowed = min_epsilon_allowed
  
  # Calculating Epsilon Decay factor. 
  epsilon_decay_factor = np.power(min_epsilon_allowed/epsilon, 1/episodes)
  
  # Initiating list to store rewards and epsilons we use across episodes
  reward_across_episodes = []
  epsilons_across_episodes = []
  
  # Iterating over Episodes.
  for _ in range(episodes):
    # Resetting the environment.
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    # Fectcing Current State and Current Action details.
    current_state = np.argmax(obs)
    current_action = agent.step(current_state, epsilon)
    
    reward_per_episode = 0
    epsilons_across_episodes.append(epsilon)
    
    # Iterating over an epsidoe untill termination status is reached.
    while not terminated:
      # Taking one step in the environment.
      obs, reward, terminated, truncated, _ = env.step(current_action)
      
      # Calculating cummulative reward for an Epoch.
      reward_per_episode = reward_per_episode+reward
      
      # Fetching future state and future reward.
      future_state = np.argmax(obs)
      future_action = agent.step(future_state, epsilon)
      
      # Updating Q values.
      agent.update_qvalue(current_state, current_action, reward, future_state)

      current_state = future_state

      current_action = future_action
    
    # Decaying Epsilon
    epsilon = epsilon_decay_factor*epsilon
    reward_across_episodes.append(reward_per_episode)
  print('\n')
  print("Trained Q-Table; {0}".format(agent.q_table))

  return agent, reward_across_episodes, epsilons_across_episodes


def doubl_q_learning_learning_loop_env(env,learning_rate: float, discount_factor: float, episodes: int,
                        min_epsilon_allowed: float, initial_epsilon_value: float) -> tuple['QLearningGreedyAgent', list, list]:
  """Learning loop train Agent to reach GOAL state in the environment using Q-Learning Algorithm.

  Args:
      env (gymnasium.Env): object of Grid Environment.
      learning_rate (float): Learning rate used in SARSA algorithm
      discount_factor (float): Discount factor to quantify the importance of future reward.
      episodes (int): Number of episodes we should train.
      min_epsilon_allowed (float): Minimum epsilon that we should reach by the end of the training.
      initial_epsilon_value (float): Initial epsilon that we should use while starting the learning. 

  Returns:
      tuple[QLearningGreedyAgent, list, list]: Returns a tuple containing agent,
                                                  cumulative rewards across episodes,
                                                  epsilon used across episodes respectively.
  """
  
  # Agent.
  agent = DoubleQLearningGreedyAgent(env, learning_rate=learning_rate, discount_factor = discount_factor)
  print("Initial Q-Table-A; {0}".format(agent.q_table_a))
  print("Initial Q-Table-B; {0}".format(agent.q_table_b))
  
  # Initiating Number of episodes, epsilon values.
  episodes = episodes

  epsilon = initial_epsilon_value
  min_epsilon_allowed = min_epsilon_allowed
  
  # Calculating Epsilon Decay factor. 
  epsilon_decay_factor = np.power(min_epsilon_allowed/epsilon, 1/episodes)
  
  # Initiating list to store rewards and epsilons we use across episodes
  reward_across_episodes = []
  epsilons_across_episodes = []
  
  # Iterating over Episodes.
  for _ in range(episodes):
    # Resetting the environment.
    obs, _ = env.reset()
    terminated, truncated = False, False
    
    # Fectcing Current State and Current Action details.
    current_state = np.argmax(obs)
    current_action = agent.step(current_state, epsilon)
    
    reward_per_episode = 0
    epsilons_across_episodes.append(epsilon)
    
    # Iterating over an epsidoe untill termination status is reached.
    while not terminated:
      # Taking one step in the environment.
      obs, reward, terminated, truncated, _ = env.step(current_action)
      
      # Calculating cummulative reward for an Epoch.
      reward_per_episode = reward_per_episode+reward
      
      # Fetching future state and future reward.
      future_state = np.argmax(obs)
      future_action = agent.step(future_state, epsilon)
      
      # Updating Q values.
      agent.update_qvalue(current_state, current_action, reward, future_state)

      current_state = future_state

      current_action = future_action
    
    # Decaying Epsilon
    epsilon = epsilon_decay_factor*epsilon
    reward_across_episodes.append(reward_per_episode)
  print('\n')
  print("Trained Q-Table-A; {0}".format(agent.q_table_a))
  print("Trained Q-Table-B; {0}".format(agent.q_table_b))

  return agent, reward_across_episodes, epsilons_across_episodes