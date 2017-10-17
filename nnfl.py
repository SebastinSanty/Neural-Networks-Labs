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
            self.layer[i].predicted_values = self.layer[i].activation.forward(values)
            values = predicted_values
            
        return predicted_values

    def backward(self, expected, predicted):
        error = expected - predicted
        for i in range (self.num,0,-1):
            for j in range(0, self.layer[i-1].units):
                self.layer[i-1].delta[j] = error[j] * self.layer[i-1].activation.backward(self.layer[i-1].predicted_values[j])
            error = np.sum(self.layer[i-1].delta * np.transpose(self.layer[i-1].weights))


    def update(self, loss):
        for i in range(self.num, 0, -1):
            for j in range(0, self.layer[i-1].units):
                self.layer[i-1].weights[j] = self.layer[i-1].weights[j] + lr*self.layer[i-1].delta*self.layer[i-2].predicted_values + mf*self.layer[i-1].del_w[j]
                self.layer[i-1].del_w[j] = lr*self.layer[i-1].delta * self.layer[i-2].predicted_values

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
        self.network.num =  network.num  + 1
        

    def train(self, X, Y):
        predicted = self.network.forward(X)
        loss = self.network.backward(Y,predicted)
        self.network.update(loss)
        return loss_value

    def predict(self, X):
        self.network.forward(X)
        return predicted_values
