### Frozen Lake Problem
Frozen lake environment in the OpenAI gym has a 4x4 grid of blocks. Each is either a start block, goal block, safe frozen block, or a dangerous hole. 

Agent's objective is to navigate from start to goal block without falling into a hole. A wind occasionally blows the agent onto a different block.

The agent can move up, down, left, or right. No reward for each step. Reward for entering the goal block is 1.

### Reinforcement Q-Learning
Reinforcement learning with Policy Networks attempts to learn functions that directly map an observation to an action. In contrast, Q-Learning attempts to learn the value of being in a given state and then takes an action.

#### Q-table learning
Simple Q-learning implementation for this problem is a table of size number of states by number of allowed actions, which is 16x4 in this game. At the start, the table of Q-values has uniform zeros, but those are updated using the Bellman equation.

Bellman equation: the expected long term reward for a given action equals the immediate reward for the current action combined with the expected reward from the best future action taken at the following state.

Q(state, action) = thisReward + discounted(max(Q(nextState, nextAction)))

#### Q-network learning
Q-table learning doesn't scale as the number of possible states grows. In lieu of table,
a network may learn to approximate a function that maps possible states to Q-values.

Simple network implementation encodes the states in a one hot vector and produces a vector of the 4 actions' Q-values. Basically this is the old table in network clothes until fancy stuff like activation functions are added. Also instead of directly updating a table, the network is updated with back propagation. The loss function is the sum-of-squares loss.
