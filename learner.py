#importing python libraries
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
import random

#importing stuff we wrote
from extract_features import extract_features
from network import Network
from mdp import transition
from replay_memory import ReplayMemory, Transition

#reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


###This class is where the main learning algorithm will be implemented.
# Idea: make simple problem (map with 1 or no obstacles), see if it works.

class Learner:
	def __init__(self):

		self.alpha = 2e-2  # Learning rate for all gradient descent methods
		self.numNodesPerLayer = [32]  # Vector of the number of hidden nodes per layer, indexed from input -> output
		self.outChannelsPerLayer = [16, 32, 32] #Number of channels per output layer (CNN)
		self.kernelSizes = [5,5,5] #size of kernel for each conv layer (CNN)
		self.K = 32  # Number of samples in our minibatch
		self.I = 20000  # Number of gradient descent iterations
		self.T = 20# Max time steps for the exploration (i.e., policy roll-out)
		self.episodes = 10000 #number of training episodes to run
		self.gamma = 0.95  # discount factor
		self.epsilon = 1 #episilon for epsilon-greedy rollouts
		self.epsilon_o = 0.05  # Minimum amount of noise to maintain in the epsilon-greedy policy rollouts
		self.epsilon_decay_const = 1e-4  # Rate at which epsilon is annealed from 1 to epsilon_O
		self.numInputDims = 5 #how many features for each state
		self.numOutputDims = 4 #how many different actions to take (move up, right, down, left)
		self.numInputChannels = 4 #Number of input channels per image
		# self.nn = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims, CNN = True, kernels = self.kernelSizes)
		self.act = [0,1,2,3] #list of actions as follows [move right 1 pixel, move down 1, move left 1
									 # move up 1, move right 2, move down 2 ...	]
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def run_learner(self):
		device = self.device
		target_net = Network(self.numInputChannels, self.outChannelsPerLayer, self.numOutputDims, 
			CNN = True, kernels = self.kernelSizes).to(device) #Q_i(s,a)
		policy_net = Network(self.numInputChannels, self.outChannelsPerLayer, 
			self.numOutputDims, CNN = True, kernels = self.kernelSizes).to(device) 
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()
		

		###IMPORTANT: In the img array indices are (y,x)
		start = (60, 120)
		goal =  (490, 440)

		map_path  = "Map_1.png"
		map = plt.imread(map_path)
		
		
		map[start, 0:3] = 0.0 #marking start on map
		plt.figure()
		imgplot = plt.imshow(map)
		plt.show()

		t_map = map.reshape(4,600,600) #t_map = torch_map, map in format for pytorch
		t_map = torch.from_numpy(t_map)
		t_map = t_map.unsqueeze(0)
		# t_map = t_map.cuda()
		print(t_map.shape)


		#Will run reinforcement learning below

		replay_buffer = ReplayMemory(10000)
		curr_pos = start #setting the current location
		curr_state = t_map #the current state is the map/env
		for i in range(self.episodes):
			print("starting Episode: {}".format(i))
			for j in range(self.T):
				sample = random.random()
				if sample < self.epsilon:
					print("Taking random action")
					action = random.choice(self.act)
				else:
					with torch.no_grad():
						q_vals = policy_net(curr_state) #q_vals of current state
					print("taking best action")
					action = (torch.argmax(q_vals)).int()
				print("action is: {}".format(action))
				r, map, next_pos = transition(map, curr_pos, goal, action)
				print("moving to: {}".format(next_pos))
				print(map[next_pos[0],next_pos[1],:])

				action = torch.tensor([[action]], device=device)
				r = torch.tensor([r], device=device)
				#updating map for torch format
				t_map = map.reshape(4,600,600) #t_map = torch_map, map in format for pytorch
				t_map = torch.from_numpy(t_map)
				t_map = t_map.unsqueeze(0)
				# t_map = t_map.cuda()

				replay_buffer.push(curr_state, action, t_map, r)
				curr_pos = next_pos
				curr_state = t_map
				policy_net  = self.optimize_model(replay_buffer, policy_net, target_net)
				self.epsilon = self.epsilon - self.epsilon_decay_const	

				
			
			target_net.load_state_dict(policy_net.state_dict()) #Q_{i+1}(s,a)
			if (i % 100 == 0):
				print("Epsilon is: {}".format(self.epsilon))
				print("size of replay_buffer is: {}".format(len(replay_buffer)))
				imgplot = plt.imshow(map)
				plt.show()


	def optimize_model(self, memory, policy_net, target_net):
	    if len(memory) < self.T:
	        return policy_net
	    transitions = memory.sample(self.T)
	    device = self.device
	    optimizer = optim.RMSprop(policy_net.parameters())
	    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	    # detailed explanation). This converts batch-array of Transitions
	    # to Transition of batch-arrays.
	    batch = Transition(*zip(*transitions))

	    # Compute a mask of non-final states and concatenate the batch elements
	    # (a final state would've been the one after which simulation ended)
	    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
	                                          batch.next_state)), device=device, dtype=torch.bool)
	    non_final_next_states = torch.cat([s for s in batch.next_state
	                                                if s is not None])
	    state_batch = torch.cat(batch.state)
	    action_batch = torch.cat(batch.action)
	    reward_batch = torch.cat(batch.reward)

	    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	    # columns of actions taken. These are the actions which would've been taken
	    # for each batch state according to policy_net
	    state_action_values = policy_net(state_batch).gather(1, action_batch)

	    # Compute V(s_{t+1}) for all next states.
	    # Expected values of actions for non_final_next_states are computed based
	    # on the "older" target_net; selecting their best reward with max(1)[0].
	    # This is merged based on the mask, such that we'll have either the expected
	    # state value or 0 in case the state was final.
	    next_state_values = torch.zeros(self.T, device=device)
	    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
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
