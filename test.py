import numpy as np
from nnfl import *

a = DenseNet(3,"momentum","svm_loss")
a.addlayer("ReLU",4)
a.addlayer("Sigmoid",1)

X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])
y1 = np.array([[0],[1],[1],[1]])
y2 = np.array([[0,0],[0,1],[1,0],[1,1]])
y3 = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])

print(a.predict(X))

print("Training..")
for i in range(100):
	error = a.train(X,y1)
	print("Epoch %d, Training Error %lf"%(i,error))

print("Testing..")

print(a.predict(np.array(X)))