import numpy as np
from nnfl import *

a = DenseNet(3,5,"a")
a.addlayer("Sigmoid",4)
a.addlayer("Sigmoid",3)

X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])
y = np.array([[0,0],[0,1],[1,0],[1,1]])
y1 = np.array([[0],[0],[1],[1]])
y3 = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])

print(a.predict(X))

print("Training..")
for i in range(1000):
	a.train(X,y3)

print("Testing..")

print(a.predict(np.array(X)))