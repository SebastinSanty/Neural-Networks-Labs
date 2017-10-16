import numpy as np

class Graph:
# Computational graph class
    def __init__(self, values_dim, optim_config, loss_fn):
        num = 0

    def addgate(self, activation, units=0):
        self.layer[num].weights = np.random(units)

        if(activation == "ReLU"):
            self.layer[num].activation = ReLU()
        elif(activation == "Sigmoid"):
            self.layer[num].activation = Sigmoid()
        elif(activation == "Softmax"):
            self.layer[num].activation = Softmax()
        elif(activation == "Linear"):
            self.layer[num].activation = Linear()

    def forward(self, values):
         
        for i in range (self.num):
            values = self.layer[i].weights * values
            predicted_values = self.layer[i].activation.forward(values)
            values = predicted_values
            
        return predicted_values

    def backward(self, expected, predicted):
        self.loss_val[self.num-1] = expected
        for i in range (self.num,0,-1):
            self.loss_val[i] = self.layer[i-1].activation.backward(self.loss_val[i-1])

    def update(self, loss):
        for i in range(self.num, 0, -1)
            layer[i-1].weights = layer[i-1].weights + 

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
        gradients_wrt_values = dz*(1-dz)
        return gradients_wrt_values

class Softmax:
# Softmax
    def __init__(self, d, m):
        pass
    def forward(self, values):
        num = np.exp(values)
        gate_output = num/(np.sum(num))
        return gate_output
    def backward(self, dz):
        gradients_wrt_valuess = dz*(1-dz)
        return gradients_wrt_values

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
        network = Graph(values_dim, optim_config, loss_fn)
               
    def addlayer(self, activation, units):
        network.addgate(activation, units)
        network.num =  network.num  + 1
        

    def train(self, X, Y):
        predicted = network.forward(X)
        loss = network.backward(Y,predicted)
        network.update(loss)
        return loss_value

    def predict(self, X):
        network.forward(X)
        return predicted_values
