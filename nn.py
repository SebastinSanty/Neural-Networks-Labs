import numpy as np

class Graph:
    # Computational graph class
    def __init__(self, input_dim, optim_config, loss_fn):
        pass

    def addgate(self, activation, units=0):
        pass

    def forward(self, input):
        return predicted_value

    def backward(self, expected):
        return loss_val

    def update(self):
        pass


class ReLU:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def relu(self, x, deriv = False):
		if(x>=0):
          rel = x
        else:
          rel = 0
		if deriv == True:
          if(x>=0):
          	return 1
          else:
			return 0
        return rel

class Sigmoid:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def sigmoid(self, x, deriv = False):
		sigm = 1/(1+np.exp(-1*x))
		if deriv == True:
			return (sigm*(1-sigm))
		return sigm

class Softmax:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def softmax(self, x, deriv = False):
      if deriv == True:
        return s*(1-sigmoid)
      return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
class Linear:
    # Example class for the ReLU layer. Replicate for other activation types and/or loss functions.
	def linear(self, x, deriv = False):
      if(x>=0):
        lin = 1
      else
      	lin = -1
        
      if deriv == True:
        return 0
      return lin
      

class DenseNet:
    def __init__(self, input_dim, optim_config, loss_fn):
        """
        Initialize the computational graph object.
        """
        pass

    def addlayer(self, activation, units):
        """
        Modify the computational graph object by adding a layer of the specified type.
        """
        pass

    def train(self, X, Y):
        """
        This train is for one iteration. It accepts a batch of input vectors.
        It is expected of the user to call this function for multiple iterations.
        """
        return loss_value

    def predict(self, X):
        """
        Return the predicted value for all the vectors in X.
        """
        return predicted_value
