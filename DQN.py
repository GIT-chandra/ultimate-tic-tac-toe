import gym, random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from uttt_env import UltTTTEnv

# Gen writing something

class DQN:	
	REPLAY_MEMORY_SIZE = 10000 			# number of tuples in experience replay  
	EPSILON = 0.9 						# epsilon of epsilon-greedy exploration
	EPSILON_DECAY = 0.999 				# exponential decay multiplier for epsilon
	HIDDEN1_SIZE = 20 					# size of hidden layer 1
	HIDDEN2_SIZE = 20 					# size of hidden layer 2
	EPISODES_NUM = 500 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 300 					# maximum number of steps in an episode 
	LEARNING_RATE = 0.01 				# learning rate and other parameters for SGD/RMSProp/Adam
	MINIBATCH_SIZE = 24 				# size of minibatch sampled from the experience replay
	DISCOUNT_FACTOR = 0.9 				# MDP's gamma
	TARGET_UPDATE_FREQ = 32 			# number of steps (not episodes) after which to update the target networks 
	SEED = 2


	# Create and initialize the environment
	def __init__(self, env):
		self.env = gym.make(env)

		# seeding
		self.env.seed(self.SEED)
		np.random.seed(self.SEED)
		random.seed(self.SEED)
		tf.set_random_seed(self.SEED)

		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)

	
	# Create the Q-network
	def initialize_network(self):
		# placeholder for the state-space input to the q-network
		self.x = tf.placeholder(tf.float32, [None, self.input_size])		
		self.y = tf.placeholder(tf.float32, [None])		# Target Q value
		# since Q value corresponding to the actions taken are to be updated,
		# only those are to be selected from self.Q. actionFilter is used for that
		self.actionFilter = tf.placeholder(tf.float32, [None, self.output_size])

		# hidden layers
		with tf.name_scope('hidden1'):
			W1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], \
						stddev=0.01), name='W1')
			b1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b1')
			h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
		with tf.name_scope('hidden2'):
			W2 = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], \
						stddev=0.01), name='W2')
			b2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b2')
			h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

		# output layer
		with tf.name_scope('output'):
			W3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], \
						stddev=0.01), name='W3')
			b3 = tf.Variable(tf.zeros(self.output_size), name='b3')
			self.Q = tf.matmul(h2, W3) + b3

		self.weights = [W1, b1, W2, b2, W3, b3]	# for use in freezing the network
		
		# computing loss
		q_values = tf.reduce_sum(tf.multiply(self.Q, self.actionFilter), 
					reduction_indices=[1])	# selecting only those values corresponding to actions
		self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.y)))

		# Training
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)

	def train(self, episodes_num=EPISODES_NUM):
		
		# Initialize the TF session
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		
		eps_dec = self.EPSILON	# epsilon with decay applied
		replay_mem = []
		total_steps = 0
		
		for episode in range(episodes_num):
			print('Episode', episode)
		  
			S = self.env.reset()	# state
			target_weights = self.session.run(self.weights)

			episode_length = 0
			while True:
				# Pick the next action using epsilon greedy and and execute it
				exploreExploit = np.random.choice(2, p = [eps_dec,1 - eps_dec])
				if exploreExploit == 0: # explore
					action = self.env.action_space.sample()
				else:   # exploit
					q_vals = self.session.run(self.Q, feed_dict={self.x: [S]})
					action = q_vals.argmax()
				
				# Step in the environment. 
				S_, R, done, _ = self.env.step(action)
				episode_length += 1
				total_steps += 1

				# Update the (limited) replay buffer. 
				replay_mem.append((S, action, R, S_, done))
				if len(replay_mem) > self.REPLAY_MEMORY_SIZE:
					replay_mem.remove(replay_mem[0])
				S = S_.copy()				
				
				# Sample a random minibatch and perform Q-learning (fetch max Q at s') 
				if len(replay_mem) <= self.MINIBATCH_SIZE:
					continue
				batch = random.sample(replay_mem, self.MINIBATCH_SIZE)

				# collection of S_
				transition_states = [sample[3] for sample in batch]
				feed_dict = {self.x: transition_states}
				feed_dict.update(zip(self.weights, target_weights))
				q_vals = self.session.run(self.Q, feed_dict=feed_dict)
				max_q_vals = q_vals.max(axis=1)

				# Creating target Q values and encoding the actions taken
				gt_q = np.zeros(self.MINIBATCH_SIZE)
				action_filter = np.zeros((self.MINIBATCH_SIZE, self.output_size), dtype=int)
				for i in range(self.MINIBATCH_SIZE):
					_, a, r, _, terminated = batch[i]
					gt_q[i] = r
					if not terminated:
						gt_q[i] += self.DISCOUNT_FACTOR * max_q_vals[i]
					action_filter[i][a] = 1

				# Training step, SGD
				states = [sample[0] for sample in batch]
				feed_dict = {
					self.x: states, 
					self.y: gt_q,
					self.actionFilter: action_filter,
				}
				_, = self.session.run([self.train_op], feed_dict=feed_dict)

			  	# Update target weights. 
				if total_steps % self.TARGET_UPDATE_FREQ == 0:
					target_weights = self.session.run(self.weights)

				# Break out of the loop if the episode ends, or is going on for too long
				if done or (episode_length == self.MAX_STEPS):
					break
			eps_dec *= self.EPSILON_DECAY

			# Logging
			print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, episode_length, total_steps))


	# Simple function to visually 'test' a policy
	def playPolicy(self):
		
		done = False
		steps = 0
		state = self.env.reset()
		
		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200: 	
			self.env.render()				
			q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			steps += 1
		
		return steps

if __name__ == '__main__':

	# Create and initialize the model
	dqn = DQN('CartPole-v0')
	dqn.initialize_network()

	print("\nStarting training...\n")
	dqn.train()
	print("\nFinished training...\nCheck out some demonstrations\n")

	# Visualize the learned behaviour for a few episodes
	results = []
	for i in range(100):
		episode_length = dqn.playPolicy()
		print("Test steps = ", episode_length)
		results.append(episode_length)
	print("Mean steps = ", sum(results) / len(results))	

	print("\nFinished.")
	plt.plot(range(len(results)), results)
	plt.xlabel('Episode')
	plt.ylabel('Length')
	plt.show()
