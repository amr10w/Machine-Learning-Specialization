import numpy as np

def my_softmax(z):
    ez=np.exp(z)
    su=ez/np.sum(ez)
    return su

z=np.array([4,3,2,1])
mult=my_softmax(z)
print(mult)

def softmax_cross_entropy_loss(Z, y):
    m,N=Z.shape
    cost=0
    
    for i in range(m):
        for j in range(N):
            if y[i]==j:
                cost+= -np.log(np.exp(Z[i][j])/np.sum(np.exp(Z[i])))
    
    return cost/m
    