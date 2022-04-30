#importing python libraries
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
import random
import gym
import numpy as np

#importing stuff we wrote
from extract_features import extract_features
from network import Network
from mdp import transition
from replay_memory import ReplayMemory, Transition
from envs.path_plan_env import PathPlanEnv

#reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


###This class is where the main learning algorithm will be implemented.
# Idea: make simple problem (map with 1 or no obstacles), see if it works.

class Learner:
	def __init__(self):

		self.alpha = 2e-3  # Learning rate for all gradient descent methods
		self.numNodesPerLayer = [32]  # Vector of the number of hidden nodes per layer, indexed from input -> output
		self.outChannelsPerLayer = [16, 32, 32] #Number of channels per output layer (CNN)
		self.kernelSizes = [5,5,5] #size of kernel for each conv layer (CNN)
		self.K = 32  # Number of samples in our minibatch
		self.I = 20000  # Number of gradient descent iterations
		self.T = 100# Max time steps for the exploration (i.e., policy roll-out)
		self.opt_iter = 10000 #max number of optimization iterations
		self.gamma = 0.95  # discount factor
		self.epsilon = 0.9 #epsilon for epsilon-greedy rollouts
		self.epsilon_o = 0.05  # Minimum amount of noise to maintain in the epsilon-greedy policy rollouts
		self.epsilon_decay_const = 1e-4	  # Rate at which epsilon is annealed from 1 to epsilon_O
		self.numInputDims = 5 #how many features for each state
		self.numOutputDims = 4 #how many different actions to take (move up, right, down, left)
		self.numInputChannels = 1 #Number of input channels per image
		# self.nn = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims, CNN = True, kernels = self.kernelSizes)
		self.act = [0,1,2,3] #list of actions as follows [move right 1 pixel, move down 1, move left 1
									 # move up 1, move right 2, move down 2 ...	]
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.batchSize = 40 #batchsize when optimizing the model
		self.totIter = 0 #number of total iterations of training/running
		self.episodeIter = 0 #number of iterations/moves per episode
		self.stepsToGoal = [] #number of steps to goal for each episode
		self.successEpis = 0 #number of episodes where the agent found the goal
		self.totalEpis = 0 #total number of episodes we have run


	def run_learner(self):
		device = self.device
		target_net = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims, 
			CNN = True, kernels = self.kernelSizes).to(device) #Q_i(s,a)
		policy_net = Network(self.numInputChannels, self.outChannelsPerLayer, 
			self.numOutputDims, CNN = True, kernels = self.kernelSizes).to(device) 
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()
		# optimizer = optim.RMSprop(policy_net.parameters())
		optimizer = optim.Adam(policy_net.parameters(), lr=self.alpha)


		#TODO: keep window of observations and pass into NN, each sample in window = 1 channel
		#TODO: larger step sizes
		###IMPORTANT: In the img array indices are (y,x)
		start = [120, 120]
		goal =  [450, 440]

		env: PathPlanEnv = gym.make("envs/PathPlanEnv-v0", file="maps/Map_7_obs.png", start=np.array(start), goal=np.array(goal))

		obs = env.reset()
		plt.imshow(obs["map"], cmap="gray")
		plt.show()

		print(obs["map"].shape)


		# t_map = map.reshape(4,600,600) #t_map = torch_map, map in format for pytorch
		t_map, curr_state, curr_pos = self.reset_vars(env)


		#Will run reinforcement learning below

		replay_buffer = ReplayMemory(10000) #this will hold a buffer of <s,a,r,s'>

		for i in range(self.opt_iter):

			for j in range(self.T):
				sample = random.random()
				if sample < self.epsilon:
					# print("Taking random action")
					action = random.choice(self.act)
				else:
					with torch.no_grad():
						q_vals = policy_net(curr_state) #q_vals of current state

					# print("taking best action")
					action = (torch.argmax(q_vals)).int().item()

				# print("action is: {}".format(action))
				# r, map, next_pos = transition(map, curr_pos, goal, action)
				obs, reward, done, _ = env.step(action)

				action = torch.tensor([[action]], device=device, dtype=torch.int64)
				reward = torch.tensor([reward], device=device)
				#updating map for torch format
				# t_map = map.reshape(4,600,600) #t_map = torch_map, map in format for pytorch
				t_map = torch.from_numpy(obs["map"] / 255)
				t_map = t_map.float()
				t_map = t_map.unsqueeze(0)
				t_map = t_map.unsqueeze(0)


				replay_buffer.push(curr_state, action, t_map, reward)
				curr_pos = env.current_position
				curr_state = t_map
				
				self.epsilon = self.epsilon - self.epsilon_decay_const
				self.totIter += 1
				self.episodeIter += 1
				if (done):
					break

			if (done):
				print("We Reached the goal. Total number of iterations: {}".format(self.episodeIter))
				self.stepsToGoal.append(self.episodeIter)
				self.successEpis += 1
				if (self.successEpis > 50):
					break
				obs = env.reset()
				t_map, curr_state, curr_pos = self.reset_vars(env)
				self.totalEpis += 1
				print("starting Episode: {}".format(self.totalEpis))

			if (self.episodeIter > 6000):
				print("Agent failed to find goal")
				t_map, curr_state, curr_pos = self.reset_vars(env)
				self.totalEpis += 1
				print("starting Episode: {}".format(self.totalEpis))

			policy_net  = self.optimize_model(replay_buffer, policy_net, target_net, optimizer)

			
			if (i % 5 == 0):
				target_net.load_state_dict(policy_net.state_dict()) #Q_{i+1}(s,a)

		plt.clf()
		plt.plot(self.stepsToGoal)
		plt.xlabel("Episode")
		plt.ylabel("Steps to goal")
		plt.xlim(0, 53)
		plt.title("lr: {} ,gamma: {}, eps-decay: {}".format(self.alpha, self.gamma, self.epsilon_decay_const))
		plt.show()


	###This function resets a few variables needed for the training loop
	def reset_vars(self,env):
		self.episodeIter = 0
		obs = env.reset()
		t_map = torch.from_numpy(obs["map"] / 255)
		t_map = t_map.float()
		t_map = t_map.unsqueeze(0)
		t_map = t_map.unsqueeze(0)
		curr_pos = env.current_position
		curr_state = t_map
		return t_map, curr_state, curr_pos


	def optimize_model(self, memory, policy_net, target_net, optimizer):
		if len(memory) < self.batchSize:
			return policy_net
		transitions = memory.sample(self.batchSize)
		device = self.device
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		# Currently not needed, might need if we change the environment
		# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
		#                                       batch.next_state)), device=device, dtype=torch.bool)
		# non_final_next_states = torch.cat([s for s in batch.next_state
		#                                             if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		next_state_batch = torch.cat(batch.next_state)


		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		# next_state_values = torch.zeros(self.T, device=device)
		# next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

		next_state_values = target_net(next_state_batch).max(1)[0].detach() #we compute Q_vals for the next state and pick the max

		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch
		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		for param in policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()
		return policy_net


if __name__ == '__main__':

	test = Learner()
	test.run_learner()
