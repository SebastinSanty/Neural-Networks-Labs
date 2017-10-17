import numpy as np
from nnfl import *

a = DenseNet(4,5,"a")
a.addlayer("ReLU",3)

print(a.predict([1,2,3,4]))