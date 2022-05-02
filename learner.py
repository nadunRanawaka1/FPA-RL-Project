#importing python libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
import random
import gym
import numpy as np
import time
from collections import deque

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
	def __init__(self, useWin, maxSuccessEpis):
		"""

		:param useWin: bool - whether or not to use a window of observations on the forward pass
		:param maxSuccessEpis: int - how many successful episodes to run before terminating
		"""

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
		if (useWin):
			self.numInputChannels = 4 #Number of input channels per image
		else:
			self.numInputChannels = 1
		# self.nn = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims, CNN = True, kernels = self.kernelSizes)
		self.act = [0,1,2,3] #list of actions as follows [move right 1 pixel, move down 1, move left 1
									 # move up 1, move right 2, move down 2 ...	]
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.batchSize = 40 #batchsize when optimizing the model
		self.totIter = 0 #number of total iterations of training/running
		self.episodeIter = 0 #number of iterations/moves per episode
		self.stepsToGoal = [] #number of steps to goal for each episode
		self.successEpis = 0 #number of episodes where the agent found the goal
		self.maxSuccessEpis = maxSuccessEpis #maximum number of successful episodes to run for
		self.totalEpis = 0 #total number of episodes we have run
		self.useWin = useWin


	def run_learner(self, policy_net, num_obs, stop_threshold = 0):
		"""
		:param policy_net: object of Network class that is used to guide the policy
		:param num_obs: how many obstacles to use in the map
		:param stop_threshold: int - if a path under this threshold is found, stop the learner
		:return:
		"""


		useWin = self.useWin
		device = self.device
		policy_net.to(device)
		target_net = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims,
			CNN = True, kernels = self.kernelSizes).to(device) #Q_i(s,a)
		
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()
		# optimizer = optim.RMSprop(policy_net.parameters())
		optimizer = optim.Adam(policy_net.parameters(), lr=self.alpha)

		print("no. params",sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
		#TODO: keep window of observations and pass into NN, each sample in window = 1 channel
		#TODO: larger step sizes
		###IMPORTANT: In the img array indices are (y,x)
		start = [120, 120]
		goal =  [450, 440]

		map_path = "maps/Map_{}_obs.png".format(num_obs)

		env: PathPlanEnv = gym.make("envs/PathPlanEnv-v0", file=map_path, start=np.array(start), goal=np.array(goal))


		obs = env.reset()
		plt.imshow(obs["map"], cmap="gray")
		# plt.show()

		print(obs["map"].shape)

		# t_map = map.reshape(4,600,600) #t_map = torch_map, map in format for pytorch
		t_map, curr_state, curr_pos = self.reset_vars(env)

		if (useWin):
			obs_win = deque(maxlen=4)
			for i in range(4):
				obs_win.append(curr_state)
			curr_state = self.obs_win_to_torch(obs_win)

		#Will run reinforcement learning below

		replay_buffer = ReplayMemory(10000) #this will hold a buffer of <s,a,r,s'>
		print("starting Episode: {}".format(self.totalEpis))
		randAct = 0 #how many random actions in this episode
		bestAct = 0 #how many best actions in this episdode

		epi_r_list = [] #list of rewards for each successful episode
		epi_r = 0 #reward for current episode
		for i in range(self.opt_iter):
			for j in range(self.T):
				sample = random.random()
				if sample < self.epsilon:
					# print("Taking random action")
					action = random.choice(self.act)
					randAct += 1
				else:
					with torch.no_grad():
						q_vals = policy_net(curr_state) #q_vals of current state

					action = (torch.argmax(q_vals)).int().item()
					bestAct += 1


				obs, reward, done, _ = env.step(action)
				epi_r  +=  int(reward)

				action = torch.tensor([[action]], device=device, dtype=torch.int64)
				reward = torch.tensor([reward], device=device, dtype=torch.float)

				#updating map for torch format
				# t_map = map.reshape(4,600,600) #t_map = torch_map

				t_map = torch.from_numpy(obs["map"] / 255)
				t_map = t_map.float()
				t_map = t_map.unsqueeze(0)
				t_map = t_map.unsqueeze(0)

				if (useWin):
					obs_win.popleft()
					obs_win.append(t_map)
					next_state = self.obs_win_to_torch(obs_win)
					replay_buffer.push(curr_state, action, next_state, reward)
					curr_state = next_state
				else:
					replay_buffer.push(curr_state, action, t_map, reward)
					curr_state = t_map

				curr_pos = env.current_position
				if (self.epsilon > self.epsilon_o):
					self.epsilon = self.epsilon - self.epsilon_decay_const
				self.totIter += 1
				self.episodeIter += 1
				if (done):
					break

			if (done):
				print("We Reached the goal. Total number of steps: {}".format(self.episodeIter))
				print("Random actions: {} . Best Actions: {}".format(randAct,bestAct))
				randAct = 0
				bestAct = 0
				self.stepsToGoal.append(self.episodeIter)
				self.successEpis += 1
				epi_r_list.append(epi_r)
				epi_r = 0
				if (self.totalEpis > self.maxSuccessEpis or (self.episodeIter < stop_threshold and self.totalEpis > 10)):
					break
				obs = env.reset()
				t_map, curr_state, curr_pos = self.reset_vars(env)

				if (useWin):
					obs_win = deque(maxlen=4)
					for i in range(4):
						obs_win.append(curr_state)
					curr_state = self.obs_win_to_torch(obs_win)
				self.totalEpis += 1
				print("starting Episode: {}".format(self.totalEpis))

			if (self.episodeIter > 4000):
				print("Agent failed to find goal")
				print("Random actions: {} . Best Actions: {}".format(randAct, bestAct))
				randAct = 0
				bestAct = 0
				epi_r = 0
				self.stepsToGoal.append(self.episodeIter)
				if (self.totalEpis > self.maxSuccessEpis):
					break
				self.epsilon += 0.4
				t_map, curr_state, curr_pos = self.reset_vars(env)
				if (useWin):
					obs_win = deque(maxlen=4)
					for i in range(4):
						obs_win.append(curr_state)
					curr_state = self.obs_win_to_torch(obs_win)
				self.totalEpis += 1
				print("starting Episode: {}".format(self.totalEpis))

			policy_net  = self.optimize_model(replay_buffer, policy_net, target_net, optimizer)

			
			if (i % 3 == 0):
				target_net.load_state_dict(policy_net.state_dict()) #Q_{i+1}(s,a)
			if (self.successEpis == 20):
				for g in optimizer.param_groups:
					g['lr'] = 0.001

		self.plot_final_graph(epi_r_list)
	### This function turns an observation window to a torch format
	def obs_win_to_torch(self, obs_win):
		obs_list = list(obs_win)
		obs = np.concatenate(obs_list, axis = 1)
		obs = torch.from_numpy(obs)
		return obs

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
	
	def create_policy_network(self):
		target_net = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims, 
			CNN = True, kernels = self.kernelSizes)
		return target_net

	def plot_final_graph(self, epi_r_list):
		plt.clf()
		fig, axis = plt.subplots(1, 2)
		axis[0].plot(self.stepsToGoal)
		axis[0].set_title("steps to goal")
		axis[1].plot([i / 5 for i in epi_r_list])
		axis[1].set_title("Episode reward")
		axis[0].set_ylabel("steps to goal")
		axis[1].set_ylabel("Reward per episode")
		axis[0].set_xlabel("Episode")
		axis[1].set_xlabel("Episode")
		# axis[0].ylabel("Steps to goal")
		axis[0].set_xlim(0, self.totalEpis)
		axis[1].set_xlim(0, self.totalEpis)
		axis[1].set_ylim(-1000, 550)
		fig.suptitle("lr: {} ,gamma: {}, eps-decay: {}".format(self.alpha, self.gamma, self.epsilon_decay_const))
		plt.show()

if __name__ == '__main__':
	start = time.time()
	useWin = False
	test = Learner(useWin, 50)

	policy_net = Network(test.numInputChannels, test.outChannelsPerLayer,
			test.numOutputDims, CNN = True, kernels = test.kernelSizes).to(test.device)

	test.run_learner(policy_net, 9, 30)
	print ("total time: ", time.time()-start)