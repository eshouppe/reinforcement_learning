"""
Solve OpenAI gym Frozen Lake test problem trying different algorithms.
"""
import random
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def frzn_lk_q_tbl_lrn():
    """Q-learning tables as frozen lake problem solving approach"""
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

    return reward_list, num_episodes

#
def frzn_lk_q_ntwrk_lrn():
    """Q-learning networks as frozen lake problem solving approach"""
    # Load the environment
    env = gym.make('FrozenLake-v0')

    tf.reset_default_graph()
    # Feed forward part of network. Used to select action.
    inputsl = tf.placeholder(shape=[1, 16], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform(shape=[16, 4], minval=0, maxval=0.01))
    Qout = tf.matmul(inputsl, W)
    predict = tf.argmax(Qout, 1)

    # Sum-of-squares loss between target and predicted Q-values
    nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
    ss_loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(ss_loss)

    init = tf.initialize_all_variables()
    # Learning parameters
    y = 0.99
    e = 0.1
    num_episodes = 2000
    # Lists are total rewards and steps per episode
    jList = []
    rList = []

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # Reset environment. Get new observation.
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            while j < 99:
                j += 1
                # Greedily choose action with e chance of random action
                a, allQ = sess.run([predict, Qout], feed_dict={inputsl: np.identity(16)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                sl, r, d, _ = env.step(a[0])
                # Obtain Q' values by feeding new state thru network
                Ql = sess.run(Qout, feed_dict={inputsl: np.identity(16)[sl:sl+1]})
                # Obtain maximium Q' and set target value
                maxQl = np.max(Ql)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQl
                # Train the network
                _, Wl = sess.run([update_model, W], feed_dict={inputsl: np.identity(16)[s:s+1], nextQ: targetQ})
                rAll += r
                s = sl
                if d:
                    # Reduce chance of random action of model trains
                    e = 1.0 / ((i/50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")


#
def plot_reward_by_try():
    """Plot the average reward by try"""
    plt.plot(range(len(all_avg_rewards)), all_avg_rewards)
    plt.ylim(0, 1)
    plt.xlabel("Try number")
    plt.ylabel("Average reward")
    plt.show()


# Execute multiple tries of algorithm and display results
# all_avg_rewards = []
# for new_try in range(10):
#     list_reward, episode_cnt = frzn_lk_q_tbl_lrn()
#     average_reward = sum(list_reward) / float(episode_cnt)
#     print("Average reward: {}".format(average_reward))
#     all_avg_rewards.append(average_reward)

# print("Best average reward in 10 tries: {}".format(np.max(all_avg_rewards)))
# plot_reward_by_try()

frzn_lk_q_ntwrk_lrn()
