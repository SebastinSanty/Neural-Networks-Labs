import numpy as np

class Graph:
# Computational graph class
    def __init__(self, values_dim, optim_config, loss_fn):
        self.num = 0
        self.layers = list()
        self.values_dim = values_dim
        self.addgate("Linear",units=self.values_dim)        


    def addgate(self, activation, units=0):

        layer = Layer()

        if(activation == "ReLU"):
            act_layer = ReLU(0,0)
        elif(activation == "Sigmoid"):
            act_layer = Sigmoid(0,0)
        elif(activation == "Softmax"):
            act_layer= Softmax(0,0)
        elif(activation == "Linear"):
            act_layer = Linear(0,0)
        else:
            return

        if self.num == 0:
            prev_units = 1
            layer.weights = np.ones((prev_units,units))

        else:
            prev_units = self.layers[self.num-1].weights.shape[1]
            layer.weights = np.random.rand(prev_units,units)

        layer.activation = act_layer

        self.layers.append(layer)
        self.num =  self.num + 1


    def forward(self, values):

        for i in range (self.num):
            print(i)
            values = np.transpose(self.layers[i].weights)*values
            self.layers[i].predicted_values = self.layers[i].activation.forward(values)
            values = self.layers[i].predicted_values
            
        return self.layers[self.num-1].predicted_values

    def backward(self, expected, predicted):
        error = expected - predicted
        for i in range (self.num,0,-1):
            for j in range(0, self.layers[i-1].units):
                self.layers[i-1].delta[j] = error[j] * self.layers[i-1].activation.backward(self.layers[i-1].predicted_values[j])
            error = np.sum(self.layers[i-1].delta * np.transpose(self.layers[i-1].weights))


    def update(self, loss):
        for i in range(self.num, 0, -1):
            for j in range(0, self.layers[i-1].units):
                self.layers[i-1].weights[j] = self.layers[i-1].weights[j] + lr*self.layers[i-1].delta*self.layers[i-2].predicted_values + mf*self.layers[i-1].del_w[j]
                self.layers[i-1].del_w[j] = lr*self.layers[i-1].delta * self.layers[i-2].predicted_values

class Layer(dict):
    pass

class ReLU:
# ReLU
    def __init__(self, d, m):
        pass
    def forward(self, values):
        if(values > 0):
            return values
        return 0
    def backward(self, dz):
        if(dz > 0):
            return 1
        return 0

class Sigmoid:
# Sigmoid
    def __init__(self, d, m):
        pass
    def forward(self, values):
        gate_output = (1.0/(1+np.exp(-values)))
        return gate_output
    def backward(self, dz):
        gradients = dz*(1-dz)
        return gradients

class Softmax:
# Softmax
    def __init__(self, d, m):
        pass
    def forward(self, values):
        num = np.exp(values)
        gate_output = num/(np.sum(num))
        return gate_output
    def backward(self, dz):
        gradients = dz*(1-dz)
        return gradients

class Linear:
# Linear
    def __init__(self, d, m):
        pass
    def forward(self, values):
        return values
    def backward(self, dz):
        return 1



class DenseNet:
    def __init__(self, values_dim, optim_config, loss_fn):
        self.network = Graph(values_dim, optim_config, loss_fn)
               
    def addlayer(self, activation, units):
        self.network.addgate(activation, units)


    def train(self, X, Y):
        predicted = self.network.forward(X)
        loss = self.network.backward(Y,predicted)
        self.network.update(loss)
        return loss_value

    def predict(self, X):
        predicted = self.network.forward(X)
        return predicted
