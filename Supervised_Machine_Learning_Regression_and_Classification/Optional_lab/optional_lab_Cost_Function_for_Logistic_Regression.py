import numpy as np
import matplotlib.pyplot as plt

x_train=np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train= np.array([0, 0, 0, 1, 1, 1])

def compute_cost_logistic(X,y,w,b):
    m=X.shape[0]
    cost=0
    for i in range(m):
        z_i=np.dot(w,X[i]) +b
        f= 1/(1+np.exp(-z_i))
        loss=-y[i]*np.log(f) -(1-y[i]) * np.log(1-f)
        cost=cost+ loss
    cost=cost/m
    return cost

w_tmp=np.arw_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(x_train, y_train, w_tmp, b_tmp))

