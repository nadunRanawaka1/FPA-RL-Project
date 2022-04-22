from network import Network
from extract_features import extract_features
import matplotlib



###This class is where the main learning algorithm will be implemented.
# Idea: make simple problem (map with 1 or no obstacles), see if it works.
class learner:
	def __init__(self):

		self.alpha = 2e-2  # Learning rate for all gradient descent methods
		self.numNodesPerLayer = [32]  # Vector of the number of hidden nodes per layer, indexed from input -> output
		self.K = 32  # Number of samples in our minibatch
		self.I = 20000  # Number of gradient descent iterations
		self.T = 100  # Max time steps for the exploration (i.e., policy roll-out)
		self.gamma = 0.95  # discount factor
		self.epsilon_o = 0.05  # Minimum amount of noise to maintain in the epsilon-greedy policy rollouts
		self.epsilon_decay_const = 1e-4  # Rate at which epsilon is annealed from 1 to epsilon_O
		self.numInputDims = 5 #how many features for each state
		self.numOutputDims = 4 #how many different actions to take (move up, right, down, left)
		self.nn = Network(self.numInputDims, self.numNodesPerLayer, self.numOutputDims)
		self.act = [0,1,2,3,4,5,6,7] #list of actions as follows [move right 1 pixel, move up 1, move down 1
									 # move left 1, move right 2, move up 2 ...	]


	def run_learner(self):
		map_path  = "Map_1.png"
		map = matplotlib.pyplot.imread(map_path)

		start = (60, 170)
		goal =  (490, 440)

		features = extract_features(map, start, start, goal) #getting features for start state

		#Will run reinforcement learning below

		tau = []

		for i in range(self.T)


