import numpy as np
w1 = np.loadtxt("linear0.csv", delimiter=',')
w2 = np.loadtxt("linear1.csv", delimiter=',')
w3 = np.loadtxt("linear2.csv", delimiter=',')
w4 = np.loadtxt("linear3.csv", delimiter=',')
w = np.stack([w1, w2, w3, w4])
np.savetxt("linear_weights", w.transpose() , delimiter=' ')