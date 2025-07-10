import numpy as np
import matplotlib.pyplot as plt
from coffee_data import data_x,data_y 

X,Y = data_x,data_y
print(X.shape, Y.shape)

def plt_roast(X,Y):
    Y = Y.reshape(-1,)
    colormap = np.array(['r', 'b'])
    fig, ax = plt.subplots(1,1,)
    ax.scatter(X[Y==1,0],X[Y==1,1], s=70, marker='x', c='red', label="Good Roast" )
    ax.scatter(X[Y==0,0],X[Y==0,1], s=100, marker='o', facecolors='none', 
               edgecolors='blue',linewidth=1,  label="Bad Roast")
    tr = np.linspace(175,260,50)
    ax.plot(tr, (-3/85) * tr + 21, color='blue',linewidth=1)
    ax.axhline(y=12,color='blue',linewidth=1)
    ax.axvline(x=175,color='blue',linewidth=1)
    ax.set_title(f"Coffee Roasting", size=16)
    ax.set_xlabel("Temperature \n(Celsius)",size=12)
    ax.set_ylabel("Duration \n(minutes)",size=12)
    ax.legend(loc='upper right')
    plt.show()
    
plt_roast(X,Y)

def sigmoid(z):
    return 1/(1+np.exp(-z))

g=sigmoid

def my_dense(a_in, W, b):
    units=W.shape[1]
    a_out=np.zeros(units)
    
    for j in range(units):
        w = W[:,j]
        z=np.dot(w,a_in)+b[j]
        a_out[j]=g(z)       
        
    return(a_out)

def my_sequential(x, W1, b1, W2, b2):
    a1=my_dense(x,W1,b1)
    a2=my_dense(a1,W2,b2)
    return a2
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def my_predict(X, W1, b1, W2, b2):
    m=X.shape[0]
    p=np.zeros((m,1))
    for i in range(m):
        p[i,0]=my_sequential(X[i],W1,b1,W2,b2)
    return p
X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example

predictions = my_predict(X_tst, W1_tmp, b1_tmp, W2_tmp, b2_tmp)


yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

## Matmul

X=np.array([[200,117]]) ## 2D array 1*2
W=np.array([[1,-3,5],
           [-2,4,-6]]) 

B=np.array([[-1,1,2]])

def dense_2(A_in,W,B):
    Z=np.matmul(A_in,W)+B  #same as @ (A_in @ W) == (np.matmul(A_in,W))
    A_out=g(Z)
    return A_out

print(dense_2(X,W,B))
