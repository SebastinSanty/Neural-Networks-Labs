import numpy as np
from nnfl import *

a = DenseNet(2,5,"a")
a.addlayer("Sigmoid",1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

x1 = np.array([[0,0]])
y1 = np.array([[0]])

print(a.predict(x1))

print("Training..")
for i in range(100):
	a.train(x1,y1)

print("Testing..")

print(a.predict(np.array(x1)))