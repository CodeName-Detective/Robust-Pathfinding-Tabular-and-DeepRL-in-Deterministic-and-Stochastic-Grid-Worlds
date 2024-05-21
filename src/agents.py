import os
import pickle
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from tqdm.notebook import tqdm_notebook as tqdm

if torch.backends.mps.is_available():
    DEVICE = torch.device(device="mps")
elif torch.cuda.is_available():
    DEVICE = torch.device(device="cuda")
else:
    DEVICE = torch.device(device="cpu")

class RandomAgent:
  def __init__(self: 'RandomAgent', env) -> None:
    """
    Initiating Random Agent
    Args:
          env (GridEnvironment): object of Grid Environment

    """
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
  
  def step(self: 'RandomAgent', state: np.ndarray) -> int:
    """
    Agent returns the action given the state of the environment

      Args:
          state (np.ndarray): Current state of the environment.

      Returns:
          action (int): action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
      
    """
    # As it is a random agent the state of the agent has no role in deciding on the action it want to take.
    return np.random.choice(self.action_space.n)


class SarsaEpsilonGreedyAgent:
  """
  Define an RL agent which follows an epsilon greedy algorithm throughout the training.
  It uses SARSA algorithm to update Q-values.
  """
  def __init__(self: 'SarsaEpsilonGreedyAgent', env, learning_rate: 'float', discount_factor: 'float') -> None:
    """Initializing Epsilon Greedy Agent

    Args:
        env (GridEnvironment): object of Grid Environment.
        learning_rate (float): Learning rate used in SARSA algorithm.
        discount_factor (float): Discount factor to quantify the importance of future reward.
    """
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    
    # Initiating the Q-table with all Zeros.
    self.q_table = np.zeros((self.observation_space.n, self.action_space.n))


  def step(self: 'SarsaEpsilonGreedyAgent', state: int, epsilon: float) -> int:
    """Given the current state and probability of choosing a random action we will
    provide the action we should choose.

    Args:
        state (int): Current location of the agent in the environment.
        epsilon (float): Probability of taking random action.

    Returns:
        action (int): Action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
    """
    # Probability Mass Function defined for the type of action we should choose
    type_of_actions = ['Random', 'Greedy']
    type_of_action_probability_distribution = [epsilon, 1-epsilon]
    
    # Sampling the type of action we should choose based on the above defined Probability Mass Function.
    action_to_choose = np.random.choice(a = type_of_actions, size=1, p=type_of_action_probability_distribution)
    
    # Choosing action
    if  action_to_choose == 'Random':
      action = np.random.choice(a = self.action_space.n)
    elif action_to_choose == 'Greedy':
      action = np.argmax(self.q_table[state,:])
    
    return action
  
  def update_qvalue(self: 'SarsaEpsilonGreedyAgent', current_state: int, current_action: int, 
                    reward: int, future_state: int, future_action: int) -> None:
    """Update the Q value based on the SARSA algorithm

    Args:
        current_state (int): Current State represented as integer
        current_action (int): Current action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
        reward (int): Immediate reward that was recieved after taking the current action.
        future_state (int): Future State represented as integer
        future_action (int): Future action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
    """
    # SARSA update Q table
    self.q_table[current_state,current_action] = self.q_table[current_state,current_action] + self.learning_rate*(reward + 
                                                                                  self.discount_factor*(self.q_table[future_state,future_action]) -
                                                                                  self.q_table[current_state,current_action]
                                                                                  )


class QLearningGreedyAgent(SarsaEpsilonGreedyAgent):
    """
    Define an RL agent which follows an epsilon greedy algorithm throughout the training.
    It uses Q-Learning algorithm to update Q-values.
    """
    def __init__(self: 'QLearningGreedyAgent', env, learning_rate: float, discount_factor: float) -> None:
        super(QLearningGreedyAgent, self).__init__(env, learning_rate, discount_factor)
    
    def update_qvalue(self: 'QLearningGreedyAgent', current_state: int, current_action: int, 
                    reward: int, future_state: int) -> None:
        """Update the Q value based on the Q Learning algorithm

        Args:
            current_state (int): Current State represented as integer
            current_action (int): Current action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
            reward (int): Immediate reward that was recieved after taking the current action.
            future_state (int): Future State represented as integer
        """
        # Q_learning update Q table
        self.q_table[current_state,current_action] = self.q_table[current_state,current_action] + self.learning_rate*(reward + 
                                                                                  self.discount_factor*(np.max(self.q_table[future_state,:])) -
                                                                                  self.q_table[current_state,current_action]
                                                                                  )

class NStepSarsaGreedyAgent(SarsaEpsilonGreedyAgent):
    """
    Define an RL agent which follows an epsilon greedy algorithm throughout the training.
    It uses N-Step Sarsa algorithm to update Q-values.
    """
    def __init__(self: 'NStepSarsaGreedyAgent', env, learning_rate: float, discount_factor: float) -> None:
        super(NStepSarsaGreedyAgent, self).__init__(env, learning_rate, discount_factor)
    
    def update_qvalue(self: 'NStepSarsaGreedyAgent', current_state: int, current_action: int, 
                    reward_return: float) -> None:
        """Update the Q value based on the N-step SARSA algorithm

        Args:
            current_state (int): Current State represented as integer
            current_action (int): Current action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
            reward_return (float): Discounted Reward Return calculated for N-steps.
        """
        # N-step SARSA update Q table
        self.q_table[current_state,current_action] = self.q_table[current_state,current_action] + self.learning_rate*(reward_return -
                                                                                  self.q_table[current_state,current_action]
                                                                                  )

class DoubleQLearningGreedyAgent:
  """
  Define an RL agent which follows an epsilon greedy algorithm throughout the training.
  It uses Double Q-Learning algorithm to estimate Q-values.
  """
  def __init__(self: 'DoubleQLearningGreedyAgent', env, learning_rate: 'float', discount_factor: 'float') -> None:
    """Initializing Epsilon Greedy Agent

    Args:
        env (gymnasium.Env): object of Grid Environment.
        learning_rate (float): Learning rate used in SARSA algorithm.
        discount_factor (float): Discount factor to quantify the importance of future reward.
    """
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    
    # Initiating the Q-table with all Zeros.
    self.q_table_a = np.zeros((self.observation_space.n, self.action_space.n))
    self.q_table_b = np.zeros((self.observation_space.n, self.action_space.n))


  def step(self: 'DoubleQLearningGreedyAgent', state: int, epsilon: float) -> int:
    """Given the current state and probability of choosing a random action we will
    provide the action we should choose.

    Args:
        state (int): Current location of the agent in the environment.
        epsilon (float): Probability of taking random action.

    Returns:
        action (int): Action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
    """
    # Epsilon-Greedy Action Selection
    
    random_number = np.random.rand()

    if random_number <= epsilon:
        action = np.random.choice(a = self.action_space.n)
    else:
        action = np.argmax((self.q_table_a[state,:]+self.q_table_b[state,:])/2)
    
    return action
  
  def update_qvalue(self: 'DoubleQLearningGreedyAgent', current_state: int, current_action: int, 
                    reward: int, future_state: int) -> None:
    """Update the Q value based on the Q-Learning algorithm

    Args:
        current_state (int): Current State represented as integer
        current_action (int): Current action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)
        reward (int): Immediate reward that was recieved after taking the current action.
        future_state (int): Future State represented as integer
    """
    # Double Q-Learning update Q table
    random_number = np.random.rand()

    if random_number <= 0.5:
        self.q_table_a[current_state,current_action] = self.q_table_a[current_state,current_action] + self.learning_rate*(reward + 
                                                                                  self.discount_factor*(self.q_table_b[future_state,np.argmax(self.q_table_a[future_state,:])]) -
                                                                                  self.q_table_a[current_state,current_action]
                                                                                  )
    else:
        self.q_table_b[current_state,current_action] = self.q_table_b[current_state,current_action] + self.learning_rate*(reward + 
                                                                                  self.discount_factor*(self.q_table_a[future_state,np.argmax(self.q_table_b[future_state,:])]) -
                                                                                  self.q_table_b[current_state,current_action]
                                                                                  )
        



class DQNAgent():
    def __init__(self, env, env_name, ValueFunction, convergence_value, learning_rate, batch_size, discount_factor, buffer_size, convergence_len = 20, initial_epsilon_value=1, min_epsilon_allowed=0.01, tau = 0.005):
        
        self.env = env
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon_value
        self.min_epsilon_allowed = min_epsilon_allowed
        self.env_name = env_name
        self.convergence_value = convergence_value
        self.convergence_len = convergence_len
        self.episode_reward_tracker = deque(maxlen=self.convergence_len)
        self.tau = tau
        self.buffer_size = buffer_size
        self.observation_space = self.env.observation_space.shape[0] if self.env.observation_space.shape else self.env.observation_space.n
        self.action_space = self.env.action_space.n
        
        self.replay_buffer = {}
        self.transistion_schema = ('observation', 'action', 'new_observation', 'reward', 'terminated')
        for field in self.transistion_schema:
            self.replay_buffer[field] = deque(maxlen=buffer_size)
        
        self.minimum_buffer_size = 10+self.batch_size
        
        self.q_value_function = ValueFunction(self.observation_space, self.action_space).to(DEVICE)
        
        # Initialize target action-value function Q^ with weights theta^ =  theta
        self.target_value_function = ValueFunction(self.observation_space, self.action_space).to(DEVICE)
        self.target_value_function.load_state_dict(self.q_value_function.state_dict())
        self.optimizer = torch.optim.Adam(self.q_value_function.parameters(), self.learning_rate)
        
        self.loss_criterion = torch.nn.MSELoss()
        
        self.buffer_counter = 0
        
        #self._delete_files_in_folder()
    
    def _delete_files_in_folder(self):
        # Get the list of files in the folder
        file_list = os.listdir('solved_images_1/'+self.env_name+'/train/')
        # Iterate through the files and delete them
        for file_name in file_list:
            file_path = os.path.join('solved_images_1/'+self.env_name+'/train/', file_name)
            os.remove(file_path)

    def _save_image(self, step, rgb_array, episode):
        fig, ax = plt.subplots()
        ax.imshow(np.asarray(rgb_array))
        # Set the title with the step number
        ax.set_title(f"Cart-Pole Training Episode: {episode}")
        plt.savefig(f'solved_images_1/'+self.env_name+'/train/{step}.jpeg')
        plt.close(fig)
    
    def _render(self, episode, step):
        # Plotting
        clear_output(wait=True)
        plt.figure(figsize=(15, 10))
        # Plot Cummulative Reward
        plt.subplot(2, 2, 1)
        plt.plot(self.reward_across_episodes)
        plt.title('Reward Per Episode at Episode: {}'.format(episode+1))
        plt.xlabel('Episode')
        plt.ylabel('Cummulative Reward')
        
        # Plot Average Loss Per Episode
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_across_episodes)
        plt.title('Loss Per Episode at Episode: {}'.format(episode+1))
        plt.xlabel('Episode')
        plt.ylabel('Average Loss Per Episode')
        
        # Plot Epsilon Decay
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilons_across_episodes)
        plt.title('Epsilon Decay at Episode: {}'.format(episode+1))
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        # Plot RGB array
        plt.subplot(2, 2, 4)
        rgb_array = self.env.render()
        self._save_image(step, rgb_array, episode)
        plt.imshow(rgb_array)
        plt.axis('off')
        plt.title('Step: {}'.format(step))
        
        plt.tight_layout()
        plt.show()

    def _get_action(self, observation):
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = torch.argmax(self.q_value_function.act(observation), dim=1).item()
        return action
    
    def _add_transistion(self, transistion):
        self.buffer_counter += 1
        for idx, field in enumerate(self.transistion_schema):
            self.replay_buffer[field].append(transistion[idx])
    
    def _get_batch_sample(self):
        # Sample the Relay Buffer
        len_buffer = len(self.replay_buffer['observation'])
        sample_batch = random.sample(range(len_buffer), self.batch_size)
        
        # Get and Prepare sample 
        self.observations = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[0]])[sample_batch], dtype=torch.float32).to(DEVICE)
        self.actions = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[1]])[sample_batch], dtype=torch.long).unsqueeze(1).to(DEVICE)
        self.new_observations =  torch.tensor(np.array(self.replay_buffer[self.transistion_schema[2]])[sample_batch], dtype=torch.float32).to(DEVICE)
        self.rewards = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[3]])[sample_batch], dtype=torch.long).unsqueeze(1).to(DEVICE)
        self.terminated_status = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[4]])[sample_batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
    def _synchronize(self):
        target_net_state_dict = self.target_value_function.state_dict()
        q_net_state_dict = self.q_value_function.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        
        self.target_value_function.load_state_dict(target_net_state_dict)

    def _learn(self):
        if self.minimum_buffer_size >= self.buffer_counter:
            return
        
        self._get_batch_sample()
        q_values = self.q_value_function(self.observations).gather(1, self.actions)
        
        with torch.inference_mode():
            max_target_q_values, _ = torch.max(self.target_value_function(self.new_observations), dim=1, keepdim=True)
        
        target_q_values = self.rewards + ((1-self.terminated_status) * (self.discount_factor *  max_target_q_values))
        loss = self.loss_criterion(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _save_model(self):
        with open(f'sadapala_{self.assignment_part}_dqn_{self.env_name}.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def trainer(self, episodes):
        epsilon_decay_factor = np.power(self.min_epsilon_allowed/self.epsilon, 1/episodes)
        self.reward_across_episodes = []
        step = 0
        for episode in tqdm(range(episodes)):
            observation = self.env.reset()[0]
            terminated = False
            self.reward_per_episode = 0
            #self._render(episode, step)
            while not terminated:
                step += 1
                action = self._get_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(action)
                terminated = terminated or truncated
                self.reward_per_episode += reward
                
                transistion = (observation, action, new_observation, reward, terminated)
                self._add_transistion(transistion)
                
                self._learn()
                self._synchronize()
                observation = new_observation
            
            self.epsilon = epsilon_decay_factor*self.epsilon
            self.reward_across_episodes.append(self.reward_per_episode)
            self.episode_reward_tracker.append(self.reward_per_episode)
            if np.mean(self.episode_reward_tracker) >= self.convergence_value:
                self._save_model()
                break
            self._save_model()




class DoubleDQNAgent():
    def __init__(self, env, env_name, ValueFunction, convergence_value, learning_rate, batch_size, discount_factor, buffer_size, convergence_len = 20, initial_epsilon_value=1, min_epsilon_allowed=0.01, tau = 0.005):
        
        self.env = env
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon_value
        self.min_epsilon_allowed = min_epsilon_allowed
        self.env_name = env_name
        self.convergence_value = convergence_value
        self.convergence_len = convergence_len
        self.episode_reward_tracker = deque(maxlen=self.convergence_len)
        self.tau = tau
        self.buffer_size = buffer_size
        self.observation_space = self.env.observation_space.shape[0] if self.env.observation_space.shape else self.env.observation_space.n
        self.action_space = self.env.action_space.n
        
        self.replay_buffer = {}
        self.transistion_schema = ('observation', 'action', 'new_observation', 'reward', 'terminated')
        for field in self.transistion_schema:
            self.replay_buffer[field] = deque(maxlen=buffer_size)
        
        self.minimum_buffer_size = 10+self.batch_size
        
        self.q_value_function = ValueFunction(self.observation_space, self.action_space).to(DEVICE)
        
        # Initialize target action-value function Q^ with weights theta^ =  theta
        self.target_value_function = ValueFunction(self.observation_space, self.action_space).to(DEVICE)
        self.target_value_function.load_state_dict(self.q_value_function.state_dict())
        self.optimizer = torch.optim.Adam(self.q_value_function.parameters(), self.learning_rate)
        
        self.loss_criterion = torch.nn.MSELoss()
        
        self.buffer_counter = 0
        
        #self._delete_files_in_folder()
    
    def _delete_files_in_folder(self):
        # Get the list of files in the folder
        file_list = os.listdir('solved_images_1/'+self.env_name+'/train/')
        # Iterate through the files and delete them
        for file_name in file_list:
            file_path = os.path.join('solved_images_1/'+self.env_name+'/train/', file_name)
            os.remove(file_path)

    def _save_image(self, step, rgb_array, episode):
        fig, ax = plt.subplots()
        ax.imshow(np.asarray(rgb_array))
        # Set the title with the step number
        ax.set_title(f"Cart-Pole Training Episode: {episode}")
        plt.savefig(f'solved_images_1/'+self.env_name+'/train/{step}.jpeg')
        plt.close(fig)
    
    def _render(self, episode, step):
        # Plotting
        clear_output(wait=True)
        plt.figure(figsize=(15, 10))
        # Plot Cummulative Reward
        plt.subplot(2, 2, 1)
        plt.plot(self.reward_across_episodes)
        plt.title('Reward Per Episode at Episode: {}'.format(episode+1))
        plt.xlabel('Episode')
        plt.ylabel('Cummulative Reward')
        
        # Plot Average Loss Per Episode
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_across_episodes)
        plt.title('Loss Per Episode at Episode: {}'.format(episode+1))
        plt.xlabel('Episode')
        plt.ylabel('Average Loss Per Episode')
        
        # Plot Epsilon Decay
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilons_across_episodes)
        plt.title('Epsilon Decay at Episode: {}'.format(episode+1))
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        # Plot RGB array
        plt.subplot(2, 2, 4)
        rgb_array = self.env.render()
        self._save_image(step, rgb_array, episode)
        plt.imshow(rgb_array)
        plt.axis('off')
        plt.title('Step: {}'.format(step))
        
        plt.tight_layout()
        plt.show()

    def _get_action(self, observation):
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = torch.argmax(self.q_value_function.act(observation), dim=1).item()
        return action
    
    def _add_transistion(self, transistion):
        self.buffer_counter += 1
        for idx, field in enumerate(self.transistion_schema):
            self.replay_buffer[field].append(transistion[idx])
    
    def _get_batch_sample(self):
        # Sample the Relay Buffer
        len_buffer = len(self.replay_buffer['observation'])
        sample_batch = random.sample(range(len_buffer), self.batch_size)
        
        # Get and Prepare sample 
        self.observations = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[0]])[sample_batch], dtype=torch.float32).to(DEVICE)
        self.actions = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[1]])[sample_batch], dtype=torch.long).unsqueeze(1).to(DEVICE)
        self.new_observations =  torch.tensor(np.array(self.replay_buffer[self.transistion_schema[2]])[sample_batch], dtype=torch.float32).to(DEVICE)
        self.rewards = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[3]])[sample_batch], dtype=torch.long).unsqueeze(1).to(DEVICE)
        self.terminated_status = torch.tensor(np.array(self.replay_buffer[self.transistion_schema[4]])[sample_batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
    def _synchronize(self):
        target_net_state_dict = self.target_value_function.state_dict()
        q_net_state_dict = self.q_value_function.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        
        self.target_value_function.load_state_dict(target_net_state_dict)

    def _learn(self):
        if self.minimum_buffer_size >= self.buffer_counter:
            return
        
        self._get_batch_sample()
        q_values = self.q_value_function(self.observations).gather(1, self.actions)
        
        with torch.inference_mode():
            q_value_action = torch.argmax(self.q_value_function(self.new_observations), dim=1).unsqueeze(1)
            max_target_q_values, _ = torch.max(self.target_value_function(self.new_observations).gather(1,q_value_action).squeeze(0), dim=1, keepdim=True)
        
        target_q_values = self.rewards + ((1-self.terminated_status) * (self.discount_factor *  max_target_q_values))
        loss = self.loss_criterion(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _save_model(self):
        with open(f'sadapala_{self.assignment_part}_ddqn_{self.env_name}.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    
    def trainer(self, episodes):
        epsilon_decay_factor = np.power(self.min_epsilon_allowed/self.epsilon, 1/episodes)
        self.reward_across_episodes = []
        step = 0
        for episode in tqdm(range(episodes)):
            observation = self.env.reset()[0]
            terminated = False
            self.reward_per_episode = 0
            #self._render(episode, step)
            while not terminated:
                step += 1
                action = self._get_action(observation)
                new_observation, reward, terminated, truncated, _ = self.env.step(action)
                terminated = terminated or truncated
                self.reward_per_episode += reward
                
                transistion = (observation, action, new_observation, reward, terminated)
                self._add_transistion(transistion)
                
                self._learn()
                self._synchronize()
                observation = new_observation
            
            self.epsilon = epsilon_decay_factor*self.epsilon
            self.reward_across_episodes.append(self.reward_per_episode)
            self.episode_reward_tracker.append(self.reward_per_episode)
            if np.mean(self.episode_reward_tracker) >= self.convergence_value:
                self._save_model()
                break
            self._save_model()   
