"""
Solve OpenAI gym Frozen Lake test problem trying different algorithms.
"""
import gym
import numpy as np


def froz_lake_q_learn():
    """Q-learning as frozen lake problem solving approach"""
    # Load the environment
    env = gym.make('FrozenLake-v0')

    # Initialize q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # Learning parameters
    learn_rate = 0.85
    discnt_rate = 0.99
    num_episodes = 2000
    # List of total rewards and steps per episode
    reward_list = []

    for idx in range(num_episodes):
        state = env.reset()
        reward_all = 0
        done = False
        j = 0

        # Q-learning algorithm
        while j < 99:
            j += 1
            # Greedily choose action by randomly picking from Q-table
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n)*(1./(idx+1)))
            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)
            # Update Q-table
            Q[state, action] = Q[state, action] + learn_rate*(
                reward + discnt_rate*np.max(Q[next_state, :]) - Q[state, action])
            reward_all += reward
            state = next_state
            if done:
                break
        reward_list.append(reward_all)

    return Q, reward_list, num_episodes


learned_q_tbl, total_reward, episode_cnt = froz_lake_q_learn()
print("Score over time: {}".format(sum(total_reward) / float(episode_cnt)))
print("Final Q-table values: ")
print(learned_q_tbl)
