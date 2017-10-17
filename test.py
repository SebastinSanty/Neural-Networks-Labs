import numpy as np
from nnfl import *

a = DenseNet(2,5,"a")
a.addlayer("Sigmoid",1)

print(a.predict(np.array([[1,2],[5,6]])))