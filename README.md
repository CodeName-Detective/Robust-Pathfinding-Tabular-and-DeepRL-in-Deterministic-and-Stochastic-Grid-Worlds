# Robust Pathfinding: Tabular and Deep Reinforcement Learning in Deterministic and Stochastic Grid Worlds

## Environment-1 (Deterministic Environemnt)

### Design
$$
\begin{array}{|c|c|c|c|}
\hline
Starting & & & \\
\hline
 & & & \\
\hline
 & & & Goal\\
\hline
\end{array}
$$

### Reward Distribution
| ↻0 | ↻0 | ↻-4 | ↻0 |
|----|----|-----|----|
| ↻0 |  0 |   0 | ↻0 |
| ↻0 | ↻0 |   3 |  10 |


### Environment Description

- **States**: The environment consists of 12 states arranged in a grid of 3 rows and 12 columns.
- **Actions**: Each state allows for 4 actions: {Right, Left, Up, Down}.
- **Starting Point**: The starting point of the environment is the top-left corner.
- **Goal Point**: The goal point is located at the bottom-right corner.

### Markov Decision Process (MDP)

The MDP includes 5 possible rewards: {-4, -1, 0, +3, +10}, distributed as follows:

- States bordering the environment: Remaining in the same state after an action yields a reward of -1.
- Goal state: Achieving the goal state results in a reward of +10.
- State at row 1, column 3: This state has a reward of -4.
- State at row 3, column 3: This state has a reward of +3.
- All other states: Assigned a reward of 0.

### Algorithms Implemented

- SARSA
- Q-Learning
- N-Step SARSA


## Environment-2 (Deterministic Transistion and Stochastic Reward)

### Design
$$
\begin{array}{|c|c|c|c| c|}
\hline
Starting & & & & \\
\hline
 & & & & \\
 \hline
 & & & & \\
\hline
 & & & & Goal\\
\hline
\end{array}
$$

### Reward Distribution
| ↻-1 | ↻-1 | ↻-4 | ↻-1 | ↻-1 |
|-----|-----|-----|-----|-----|
| ↻-1 |  2  |  0  |  0  | ↻-1 |
| ↻-1 |  0  |  0  |  1  | ↻-1 |
| ↻-1 | ↻-1 | ↻-1 |  3  | 10  |


### Grid World Environment Description

- **States**: The environment comprises 20 states arranged in a grid of 4 rows and 5 columns.
- **Actions**: Each state allows for 4 actions: {Right, Left, Up, Down}.
- **Starting Point**: The starting point of the environment is the top-left corner.
- **Goal Point**: The goal point is located at the bottom-right corner.

### Markov Decision Process (MDP)

The MDP includes 7 possible rewards: {-4, -1, 0, +1, +2, +3, +10}, distributed as follows:

- States bordering the environment: Remaining in the same state after an action yields a reward of -1.
- Goal state: Achieving the goal state results in a reward of +10.
- State at row 1, column 3: This state has a reward of -4.
- State at row 2, column 2: This state has a reward of +2.
- State at row 3, column 4: This state has a reward of +1.
- State at row 4, column 4: This state has a reward of +3.
- All other states: Assigned a reward of 0, and positive rewards vanish once collected.

### Objective

The primary objective of this RL environment is for the agent to start at the starting point and reach the goal position in the minimum number of time steps. The agent should enter the destination state from the left side, specifically entering from the top. The desired trajectory does not involve the agent reaching the destination from the topmost row.

### Algorithms Implemented

- Q-Learning
- Double Q-Learning

## Environment-3 (Stochastic Transistion and Stochastic Reward)

### Design
$$
\begin{array}{|c|c|c|c| c|}
\hline
Starting & & & & \\
\hline
 & & & & \\
 \hline
 & & & & \\
\hline
 & & & & Goal\\
\hline
\end{array}
$$

### Reward Distribution
| ↻-1 | ↻-1 | ↻-4 | ↻-1 | ↻-1 |
|-----|-----|-----|-----|-----|
| ↻-1 |  2  |  0  |  0  | ↻-1 |
| ↻-1 |  0  |  0  |  1  | ↻-1 |
| ↻-1 | ↻-1 | ↻-1 |  3  | 10  |


### Grid World Environment Description

- **States**: The environment comprises 20 states arranged in a grid of 4 rows and 5 columns.
- **Actions**: Each state allows for 4 actions: {Right, Left, Up, Down}.
- **Starting Point**: The starting point of the environment is the top-left corner.
- **Goal Point**: The goal point is located at the bottom-right corner.

### Markov Decision Process (MDP)

The MDP includes 7 possible rewards: {-4, -1, 0, +1, +2, +3, +10}, distributed as follows:

- States bordering the environment: Remaining in the same state after an action yields a reward of -1.
- Goal state: Achieving the goal state results in a reward of +10.
- State at row 1, column 3: This state has a reward of -4.
- State at row 2, column 2: This state has a reward of +2.
- State at row 3, column 4: This state has a reward of +1.
- State at row 4, column 4: This state has a reward of +3.
- All other states: Assigned a reward of 0, and positive rewards vanish once collected.

The Stochastic Environment introduces randomness into the previously defined Deterministic Environment through a constant State-Transition Probability Matrix for all states given the actions. Here is the matrix:

| Action | Move Right | Move Left | Move Up | Move Down |
|--------|------------|-----------|---------|-----------|
| Right  | 0.95       | 0.05      | 0       | 0         |
| Left   | 0.05       | 0.5       | 0       | 0         |
| Up     | 0          | 0         | 0.95    | 0.05      |
| Down   | 0          | 0         | 0.05    | 0.95      |

This matrix satisfies the condition of stochasticity as all the row sums (marginals) add up to one, indicating the probabilities of transitioning to each state given an action.

### Objective

The primary objective of this RL environment is for the agent to start at the starting point and reach the goal position in the minimum number of time steps. The agent should enter the destination state from the left side, specifically entering from the top. The desired trajectory does not involve the agent reaching the destination from the topmost row.

### Algorithms Implemented

- Q-Learning
- Double Q-Learning
- Deep Q-Learning
- Double Deep Q-Learning