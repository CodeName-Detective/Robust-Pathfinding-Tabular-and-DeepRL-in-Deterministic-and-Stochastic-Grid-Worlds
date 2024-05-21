import gymnasium
import matplotlib.pyplot as plt
import numpy as np


class GridEnvironment(gymnasium.Env):
  """
  An Grid RL environment that has 12 states(3x4 grid space) and each state has 4
  actions: {Down, Up, Left, Right} and has total 5 possible rewards: {-4, -1, 0, +3, +10}

  For the states that borders the environment, if the agent stays in the same
  state after taking the action we will get a reward of -1.
  The Goal state([2,3]) has the reward of +10
  The state [0,2] has the reward of -4
  The state [2,2] has the reward of 3

  """
  metadata = { 'render.modes': [] }

  def __init__(self: 'GridEnvironment', max_timesteps: int) -> None:

    """
    Initializes the Grid Environment

    Args:
        max_timesteps (int): Maximum number of timesteps allowed per episode
    """

    # Initializing the Observation Space and Action Space
    self.observation_space = gymnasium.spaces.Discrete(12)
    self.action_space = gymnasium.spaces.Discrete(4)

    # Initializing the max_timesteps allowed and current timestep
    self.max_timesteps = max_timesteps
    self.time_step = 0

    # Intializing the Agent Starting position and Goal Position
    self.agent_pos = [0,0]
    self.goal_pos = [2,3]

    # Initializing the State of the environment.
    self.state = np.zeros((3,4))
    self.state[tuple(self.agent_pos)] = 1
    self.state[tuple(self.goal_pos)] = 0.5

  def reset(self: 'GridEnvironment', **kwargs) -> tuple[np.ndarray, dict]:
    """
    Reset the Grid Environment

    Returns:
      tuple[np.ndarray, dict]: Returns a tuple containing observation numpy array and info dictionary respectively.
    """

    # Resetting the agent position to initial position.
    self.agent_pos = [0,0]

    # Resetting the time step.
    self.time_step = 0

    # Resetting the environment state
    self.state = np.zeros((3,4))
    self.state[tuple(self.agent_pos)] = 1
    self.state[tuple(self.goal_pos)] = 0.5

    observation = self.state.flatten()

    info = {}
    info['Termination Message'] = 'Not Terminated'

    return observation, info
  
  def step(self: 'GridEnvironment', action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
    """Take one step in the Grid Environment given the action

      Args:
          action (int): action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)

      Returns:
          tuple[np.ndarray, int, bool, bool, dict]: Returns a tuple containing observation, 
                                                    reward, termination status, truncation 
                                                    status, and info dictionary respectively
      """

    #Maintaining the agent old position.
    agent_old_pos = self.agent_pos.copy()

    if action == 0:
      # Take Down action.
      self.agent_pos[0] += 1
    elif action == 1:
      # Take Up action.
      self.agent_pos[0] -= 1
    elif action == 2:
      # Take Right action.
      self.agent_pos[1] += 1
    elif action == 3:
      # Take Left action.
      self.agent_pos[1] -= 1
    
    # Clipping the agent position to the same position if it goes
    # out of bound of the environment.
    self.agent_pos[0] = np.clip(self.agent_pos[0], 0, 2)
    self.agent_pos[1] = np.clip(self.agent_pos[1], 0, 3)

    # Updating the Environment state.
    self.state = np.zeros((3,4))
    self.state[tuple(self.goal_pos)] = 0.5
    self.state[tuple(self.agent_pos)] = 1
    observation = self.state.flatten()

    # Default reward of 0
    reward = 0

    if np.array_equal(self.agent_pos, self.goal_pos):
      # Assigining the reward of +10 on reaching the goal position.
      reward = 10
    elif np.array_equal(self.agent_pos, agent_old_pos):
      # Assigining the reward of -1 on statying the same position after action.
      reward = -1
    elif np.array_equal(self.agent_pos, [0,2]):
      # Assigining the reward of -4 on reaching the position of [0,2]
      reward = -4
    elif np.array_equal(self.agent_pos, [2,2]):
      # Assigining the reward of +3 on reaching the position of [2,2]
      reward = 3
    
    # Updating the time step.
    self.time_step += 1

    info = {}

    # Updating the episode termination status.
    if np.array_equal(self.agent_pos, self.goal_pos):
      # Goal position is reached.
      terminated = True
      info['Termination Message'] = 'Goal Position Reached !!!'
    elif self.time_step >= self.max_timesteps:
      # Maximum time steps reached.
      terminated = True
      info['Termination Message'] = 'Maximum Time Reached'
    else:
      # Episode not terminated.
      terminated = False
      info['Termination Message'] = 'Not Terminated'

    if (self.agent_pos[0] >=0) & (self.agent_pos[0] <= 2) & (self.agent_pos[1] >=0)  & (self.agent_pos[1] <= 3):
      truncated = True
    else:
      truncated = False
    

    return observation, reward, terminated, truncated, info
  

  def render(self: 'GridEnvironment'):
    """
    Render the image.
    """
    plt.imshow(self.state)


class DeterministicGridEnvironment(gymnasium.Env):
  """
  An Grid RL environment that has 20 states(4x5 grid space) and each state has 4
  actions: {Down, Up, Left, Right} and has total 7 possible rewards: {-4, -1, 0, +1, +2, +3, +10}
  
  Reward distribution:
  For the states that borders the environment, if the agent stays in the same
  state after taking the action we will get a reward of -1.
  The Goal state([3,4]) has the reward of +10
  The state [0,2] has the reward of -4
  The state [1,1] has the reward of +2
  The state [2,3] has the reward of +1
  The state [3,3] has the reward of +3

  """
  metadata = { 'render.modes': [] }

  def __init__(self: 'DeterministicGridEnvironment', max_timesteps: int) -> None:

    """
    Initializes the Grid Environment

    Args:
        max_timesteps (int): Maximum number of timesteps allowed per episode
    """

    # Initializing the Observation Space and Action Space
    self.observation_space = gymnasium.spaces.Discrete(20)
    self.action_space = gymnasium.spaces.Discrete(4)

    # Initializing the max_timesteps allowed and current timestep
    self.max_timesteps = max_timesteps
    self.time_step = 0

    # Intializing the Agent Starting position and Goal Position
    self.agent_pos = [0,0]
    self.goal_pos = [3,4]

    # Initializing the State of the environment.
    self.state = np.zeros((4,5))
    self.state[tuple(self.agent_pos)] = 1
    self.state[tuple(self.goal_pos)] = 0.5
    
    self.visited_1_1 = 0
    self.visited_2_3 = 0
    self.visited_3_3 = 0

  def reset(self: 'DeterministicGridEnvironment', **kwargs) -> tuple[np.ndarray, dict]:
    """
    Reset the Grid Environment

    Returns:
      tuple[np.ndarray, dict]: Returns a tuple containing observation numpy array and info dictionary respectively.
    """

    # Resetting the agent position to initial position.
    self.agent_pos = [0,0]

    # Resetting the time step.
    self.time_step = 0

    # Resetting the environment state
    self.state = np.zeros((4,5))
    self.state[tuple(self.agent_pos)] = 1
    self.state[tuple(self.goal_pos)] = 0.5
    
    self.visited_1_1 = 0
    self.visited_2_3 = 0
    self.visited_3_3 = 0
    
    observation = self.state.flatten()

    info = {}
    info['Termination Message'] = 'Not Terminated'

    return observation, info
  
  def step(self: 'DeterministicGridEnvironment', action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
    """Take one step in the Grid Environment given the action

      Args:
          action (int): action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)

      Returns:
          tuple[np.ndarray, int, bool, bool, dict]: Returns a tuple containing observation, 
                                                    reward, termination status, truncation 
                                                    status, and info dictionary respectively
      """

    #Maintaining the agent old position.
    agent_old_pos = self.agent_pos.copy()

    if action == 0:
      # Take Down action.
      self.agent_pos[0] += 1
    elif action == 1:
      # Take Up action.
      self.agent_pos[0] -= 1
    elif action == 2:
      # Take Right action.
      self.agent_pos[1] += 1
    elif action == 3:
      # Take Left action.
      self.agent_pos[1] -= 1
    
    # Clipping the agent position to the same position if it goes
    # out of bound of the environment.
    self.agent_pos[0] = np.clip(self.agent_pos[0], 0, 3)
    self.agent_pos[1] = np.clip(self.agent_pos[1], 0, 4)

    # Updating the Environment state.
    self.state = np.zeros((4,5))
    self.state[tuple(self.goal_pos)] = 0.5
    self.state[tuple(self.agent_pos)] = 1
    observation = self.state.flatten()

    # Default reward of 0
    reward = 0
    
    if np.array_equal(self.agent_pos, self.goal_pos):
      # Assigining the reward of +10 on reaching the goal position.
      reward = 10
    elif np.array_equal(self.agent_pos, agent_old_pos):
      # Assigining the reward of -1 on statying the same position after action.
      reward = -1
    elif np.array_equal(self.agent_pos, [0,2]):
      # Assigining the reward of -4 on reaching the position of [0,2]
      reward = -4
    elif np.array_equal(self.agent_pos, [1,1]):
      # Assigining the reward of +2 on reaching the position of [1,1]
      if self.visited_1_1 == 0:
        reward = +2
      else:
        reward = 0
      self.visited_1_1 += 1
    elif np.array_equal(self.agent_pos, [2,3]):
      # Assigining the reward of +1 on reaching the position of [2,3]
      if self.visited_2_3 == 0:
        reward = +1
      else:
        reward = 0
      self.visited_2_3 += 1
    elif np.array_equal(self.agent_pos, [3,3]):
      # Assigining the reward of +3 on reaching the position of [3,3]
      if self.visited_3_3 == 0:
        reward = +3
      else:
        reward = 0
      self.visited_3_3 += 1
    
    # Updating the time step.
    self.time_step += 1

    info = {}

    # Updating the episode termination status.
    if np.array_equal(self.agent_pos, self.goal_pos):
      # Goal position is reached.
      terminated = True
      info['Termination Message'] = 'Goal Position Reached !!!'
    elif self.time_step >= self.max_timesteps:
      # Maximum time steps reached.
      terminated = True
      info['Termination Message'] = 'Maximum Time Reached'
    else:
      # Episode not terminated.
      terminated = False
      info['Termination Message'] = 'Not Terminated'

    if (self.agent_pos[0] >=0) & (self.agent_pos[0] <= 3) & (self.agent_pos[1] >=0)  & (self.agent_pos[1] <= 4):
      truncated = True
    else:
      truncated = False
    

    return observation, reward, terminated, truncated, info
  

  def render(self: 'DeterministicGridEnvironment'):
    """
    Render the image.
    """
    plt.imshow(self.state)


class StochasticGridEnvironment(DeterministicGridEnvironment):
  """
  An Grid RL environment that has 20 states(4x5 grid space) and each state has 4
  actions: {Down, Up, Left, Right} and has total 7 possible rewards: {-4, -1, 0, +1, +2, +3, +10}
  
  Reward distribution:
  For the states that borders the environment, if the agent stays in the same
  state after taking the action we will get a reward of -1.
  The Goal state([3,4]) has the reward of +10
  The state [0,2] has the reward of -4
  The state [1,1] has the reward of +2
  The state [2,3] has the reward of +1
  The state [3,3] has the reward of +3

  """
  metadata = { 'render.modes': [] }

  def __init__(self: 'StochasticGridEnvironment', max_timesteps: int) -> None:

    """
    Initializes the Grid Environment

    Args:
        max_timesteps (int): Maximum number of timesteps allowed per episode
    """
    
    super(StochasticGridEnvironment, self).__init__(max_timesteps)
  
  def step(self: 'StochasticGridEnvironment', action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
    """Take one step in the Grid Environment given the action

      Args:
          action (int): action represented as number; (0: Down, 1: Up, 2: Right, 3: Left)

      Returns:
          tuple[np.ndarray, int, bool, bool, dict]: Returns a tuple containing observation, 
                                                    reward, termination status, truncation 
                                                    status, and info dictionary respectively
      """

    #Maintaining the agent old position.
    agent_old_pos = self.agent_pos.copy()
    
    
    ### Defining State Transistion Probability Distribution
    if action == 0:
      # Action Took Go Down
      if np.random.rand() < 0.95:
        # Take Down action with probability 0.9
        self.agent_pos[0] += 1
      else:
        # Take Up action with probability 0.1
        self.agent_pos[0] -= 1
    
    elif action == 1:
      # Action Took Go Up
      if np.random.rand() < 0.95:
        # Take Up action with probability 0.9
        self.agent_pos[0] -= 1
      else:
        # Take Down action with probability 0.1
        self.agent_pos[0] += 1
        
    elif action == 2:
      # Action Took Go Right
      if np.random.rand() < 0.95:
        # Take Right action with probability 0.9
        self.agent_pos[1] += 1
      else:
        # Take Left action with probability 0.1
        self.agent_pos[1] -= 1 
    elif action == 3:
      # Action Took Go Left
      if np.random.rand() < 0.95:
        # Take Left action with probability 0.9
        self.agent_pos[1] -= 1
      else:
        # Take Right action with probability 0.1
        self.agent_pos[1] += 1
    
    # Clipping the agent position to the same position if it goes
    # out of bound of the environment.
    self.agent_pos[0] = np.clip(self.agent_pos[0], 0, 3)
    self.agent_pos[1] = np.clip(self.agent_pos[1], 0, 4)

    # Updating the Environment state.
    self.state = np.zeros((4,5))
    self.state[tuple(self.goal_pos)] = 0.5
    self.state[tuple(self.agent_pos)] = 1
    observation = self.state.flatten()

    # Default reward of 0
    reward = 0
    
    if np.array_equal(self.agent_pos, self.goal_pos):
      # Assigining the reward of +10 on reaching the goal position.
      reward = 10
    elif np.array_equal(self.agent_pos, agent_old_pos):
      # Assigining the reward of -1 on statying the same position after action.
      reward = -1
    elif np.array_equal(self.agent_pos, [0,2]):
      # Assigining the reward of -4 on reaching the position of [0,2]
      reward = -4
    elif np.array_equal(self.agent_pos, [1,1]):
      # Assigining the reward of +2 on reaching the position of [1,1]
      if self.visited_1_1 == 0:
        reward = +2
      else:
        reward = 0
      self.visited_1_1 += 1
    elif np.array_equal(self.agent_pos, [2,3]):
      # Assigining the reward of +1 on reaching the position of [2,3]
      if self.visited_2_3 == 0:
        reward = +1
      else:
        reward = 0
      self.visited_2_3 += 1
    elif np.array_equal(self.agent_pos, [3,3]):
      # Assigining the reward of +3 on reaching the position of [3,3]
      if self.visited_3_3 == 0:
        reward = +3
      else:
        reward = 0
      self.visited_3_3 += 1
    # Updating the time step.
    self.time_step += 1

    info = {}

    # Updating the episode termination status.
    if np.array_equal(self.agent_pos, self.goal_pos):
      # Goal position is reached.
      terminated = True
      info['Termination Message'] = 'Goal Position Reached !!!'
    elif self.time_step >= self.max_timesteps:
      # Maximum time steps reached.
      terminated = True
      info['Termination Message'] = 'Maximum Time Reached'
    else:
      # Episode not terminated.
      terminated = False
      info['Termination Message'] = 'Not Terminated'

    if (self.agent_pos[0] >=0) & (self.agent_pos[0] <= 3) & (self.agent_pos[1] >=0)  & (self.agent_pos[1] <= 4):
      truncated = True
    else:
      truncated = False
    

    return observation, reward, terminated, truncated, info