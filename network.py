from torch import nn
import torch

### This class will hold the neural net. Might have to play around with the architecture
class Network(nn.Module):
	def __init__(self, numInDims, unitsPerLayer, numOutDims, CNN = False, kernels = [], lin_feat = None):

		''' 
		Creates a neural net. Inputs:

		numInputDims      	-   This number represents the cardinality of the
							input vector for the neural network
		unitsPerLayer		- array containing the number of nodes in each hidden layer
		numOutDims    	-   This number represents the cardinality of the
							output vector of the neural network
		CNN					-whether to use a CNN
		lin_feat            - if using CNN, how many input features to linear layer

		'''
		super(Network, self).__init__()
		self.layers = []
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if (CNN):
			if (len(kernels) != len(unitsPerLayer)):
				raise Exception("Number of kernel sizes and number of out channels must be equal.")
			for ind, units in enumerate(unitsPerLayer):
				self.layers.append(nn.Conv2d(numInDims, units, kernels[ind], stride=2, device=self.device))
				self.layers.append(nn.BatchNorm2d(units, device=self.device))
				self.layers.append(nn.ReLU())
				numInDims= units
			self.layers.append(nn.Linear(165888, 4, device=self.device))
		else:
			for ind, units in enumerate(unitsPerLayer):
				if (ind == 0):
					self.layers.append(nn.Linear(numInDims, units, device=self.device))
				else:
					self.layers.append(nn.Linear(unitsPerLayer[ind - 1], units, device=self.device))

			self.layers.append(nn.Linear(unitsPerLayer[-1], numOutDims, device=self.device))

		self.model = nn.Sequential(*self.layers)
		


	def forward(self, inputs):
		inputs = inputs.cuda()
		for layer in self.model:
			if isinstance(layer, nn.Linear): #handling linear layers
				inputs = layer(inputs.view(inputs.size(0),-1))
			else: #handling all other layers
				inputs = layer(inputs)

		return inputs