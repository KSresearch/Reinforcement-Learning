# Code for the reinforcement practice using open ai gym - DQN
# written by Hyun Seok, Whang
# date : 2018-11-06
# algorithm: DQN
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

env = gym.make('CartPole-v0')
env._max_episode_steps = 5000

# setting parameter
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
train_rate = 1e-1
discount_factor = 0.9
max_episode = 5000
memory_length = 50000
epsilon_end = 0.01
batch_size = 10

room = deque()
# memory buffer function
def append_sample(room, state, reward, done, next_state):
	if len(room) >= memory_length:
		room.popleft()
	room.append([state,reward,done,next_state])

# action selection function
def get_action(epsilon, q_val):
	if np.random.rand(1) <= epsilon:
		action = np.random.randint(action_size,size=1)[0]
		# action = env.action_space.sample()
	else:
		action = np.argmax(q_val)
	return action
	
# build network - main
with tf.variable_scope('Main_network'):
	main_input = tf.placeholder(shape=[None,state_size],dtype = tf.float32,name='main_in')
	main_hidden_1 = tf.layers.dense(main_input,48,kernel_initializer = tf.contrib.layers.xavier_initializer(),activation=tf.nn.tanh,name='main_h1')
	main_out_q = tf.layers.dense(main_hidden_1,action_size,name='main_q')
	objective = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
	loss = tf.reduce_mean(tf.square(main_out_q-objective))

# build network - target network
with tf.variable_scope('Target_network'):
	target_input = tf.placeholder(shape=[None,state_size],dtype = tf.float32,name='target_in')
	target_hidden_1 = tf.layers.dense(target_input,48,kernel_initializer = tf.contrib.layers.xavier_initializer(),activation=tf.nn.tanh,name='target_h1')
	target_out_q = tf.layers.dense(target_hidden_1,action_size,name='target_q')
	
vars_m = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Main_network')
optimizer = tf.train.AdamOptimizer(learning_rate = train_rate).minimize(loss,var_list = vars_m)

# train
def train(sess,batch_size,room):
	batch = random.sample(room,batch_size)
	
	states = np.zeros((batch_size,state_size))
	rewards = np.zeros(batch_size)
	dones = np.zeros(batch_size)
	next_states = np.zeros((batch_size,state_size))
	
	for i in range(batch_size):
		batch[i][0] = states[i]
		batch[i][1] = rewards[i]
		batch[i][2] = dones[i]
		batch[i][3] = next_states[i]
	
	target = sess.run(main_out_q, feed_dict={main_input:states})
	target_val = sess.run(target_out_q, feed_dict={target_input:next_states})
	
	# main network training
	for i in range(batch_size):
		if dones[i] == True:
			target[i] = rewards[i]
		else:
			target[i] = rewards[i]+discount_factor*np.max(target_val[i])
	sess.run(optimizer,feed_dict={main_input:states, objective:target})

# network update function
def get_copy_var_ops(*,dest_scope_name='Target_network',src_scope_name='Main_network'):
	#Copy variables src_scope to dest_scope
	op_holder=[]	
	src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=src_scope_name)
	dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=dest_scope_name)	
	for src_var, dest_var in zip(src_vars,dest_vars):
		op_holder.append(dest_var.assign(src_var.value()))  # line 96 means dest_var.assign(src_var.value()) is equal to 'dest_var = src_var'	
	return op_holder
	
# main code start
# Begin with tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Initialize the main network and target network
copy_ops = get_copy_var_ops(dest_scope_name='Target_network', src_scope_name='Main_network')
sess.run(copy_ops)

for episode in range(max_episode):
	eps = 1./((episode/10)+1)

	state = env.reset()
	state = np.reshape(state,[1,state_size])
	done = False
	step = 0
	total_reward = 0
	
	# step start
	while not done:
		step += 1
		# episilon decaying and get the q_value!
		q_val = sess.run(main_out_q, feed_dict={main_input:state})[0]
		act = get_action(eps, q_val)
		next_state,reward,done,_  = env.step(act)
		next_state = np.reshape(next_state,[1,state_size])		
		total_reward += reward	
		append_sample(room,state,reward,done,next_state)
		state = next_state
		
		# train every batch_size
		if step%batch_size == 0:
			train(sess,batch_size,room)
		
		# update target model every 50steps
		if step % 50 == 1: 		
			sess.run(copy_ops)
		
		if done:
			reward = -5	
			#and print out the entire result from the process
			print('Episode : %d   Total_reward : %d'%(episode,total_reward))
				
# After training process finished, rendering the cartpole question			
state = env.reset()
state = np.reshape(state,[1,state_size])
reward_sum = 0 
while True:
	env.render()
	action = np.argmax(sess.run(main_out_q,feed_dict={main_input:state}))
	state, reward, done,_ = env.step(action)
	state = np.reshape(state,[1,state_size])
	reward_sum += reward
	if done:
		print('Total score: %d'%(reward_sum))
		break