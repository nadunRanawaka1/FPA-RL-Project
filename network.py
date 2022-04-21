from torch import nn

### This class will hold the neural net. Might have to play around with the architecture
class Network(nn.Module):
	def __init__(self, numInDims, unitsPerLayer, numOutDims):

		''' 
		Creates a neural net. Inputs:

		numInputDims      	-   This number represents the cardinality of the
							input vector for the neural network
		unitsPerLayer		- array containing the number of nodes in each hidden layer
		numOutDims    	-   This number represents the cardinality of the
							output vector of the neural network

		'''

		self.layers = []

		for ind, units in enumerate(unitsPerLayer):
			if (ind == 0):
				self.layers.append(nn.Linear(numInDims, units))
			else:
				self.layers.append(nn.Linear(unitsPerLayer[ind - 1], units))

		self.layers.append(unitsPerLayer[-1], numOutDims)

		self.model = nn.Sequential(*layers)

	def forward(self, inputs):
		return self.model.forward(inputs) 