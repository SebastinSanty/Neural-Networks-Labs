import numpy as np
from nnfl import *

a = DenseNet(2,5,"a")
a.addlayer("Sigmoid",4)
a.addlayer("Sigmoid",1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])


print(a.predict(X))

print("Training..")
for i in range(1000):
	a.train(X,y)

print("Testing..")

print(a.predict(np.array(X)))