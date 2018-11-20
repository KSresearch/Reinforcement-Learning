# Reinforcement Learning - A2C cartpole python file
# Date : 2018-11-19
# Author: Hyun Seok, Whang
import numpy as np
import gym
import tensorflow as tf

env = gym.make('CartPole-v0')
env._max_episode_steps=5000
action_size = env.action_space.n
state_size  = env.observation_space.shape[0]

# setting parameters
max_episode = 1000
discount_factor = 0.99
action_lr = 0.001
critic_lr = 0.005
hidden_size = 24

# A2C network building
# build network - action network
input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
action_hidden = tf.layers.dense(input, hidden_size,activation=tf.nn.relu)
action_output = tf.layers.dense(action_hidden,action_size,activation=tf.nn.softmax)
# build network - critic network
critic_hidden = tf.layers.dense(input,hidden_size,activation=tf.nn.relu)
critic_output = tf.layers.dense(critic_hidden,1)
# Actor loss function and its optimizer
adt = tf.placeholder(shape=[None,],dtype=tf.float32)
actions = tf.placeholder(shape=[None,],dtype=tf.int32)
action_onehot = tf.one_hot(actions,action_size,dtype=tf.float32)
action_prob = tf.reduce_sum(action_onehot*action_output,axis=1)
action_loss = -tf.reduce_sum(tf.log(action_prob+1e-10)*adt)
action_optimizer = tf.train.AdamOptimizer(action_lr).minimize(action_loss)
# Critic loss function and its optimizer
tar = tf.placeholder(shape=[None,],dtype=tf.float32)
critic_loss = tf.reduce_sum(tf.square(tar-critic_output))
critic_optimizer = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

# function for the optimal action using policy
def get_action(policy):
    return np.random.choice(action_size,1,p=policy)[0]

# When learning process finish, show its result
def replay():
    total_reward = 0
    state = env.reset()
    state = np.reshape(state,[1,state_size])
    while True:
        env.render()
        prob = sess.run(action_output,feed_dict={input:state})[0]
        action = get_action(prob)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state,[1,state_size])
        total_reward += reward
        if done:
            print('Final reward is: %d'%(total_reward))
            break

# main code starts!
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for episode in range(max_episode):
    state = env.reset()
    state = np.reshape(state,[1,state_size])
    total_reward = 0
    done = False

    while not done:
        policy = sess.run(action_output,feed_dict={input:state})[0]
        value = sess.run(critic_output,feed_dict={input:state})[0]
        action = get_action(policy)
        next_state,reward,done,_ = env.step(action)
        next_state = np.reshape(next_state,[1,state_size])
        next_value = sess.run(critic_output,feed_dict={input:next_state})[0]

        if not done:
            advantage = reward+discount_factor*next_value-value
            target = reward+discount_factor*next_value
        else:
            reward = -1
            advantage = reward-value
            target = [reward]
        sess.run([action_optimizer, critic_optimizer], feed_dict={input:state,actions:[action],adt:advantage,tar:target})
        state = next_state
        total_reward += reward

        if done:
            print('Episode: %d,  Reward:  %d'%(episode+1,total_reward))


replay()
print('Entire process has finished!')