import numpy as np
import math,copy
import matplotlib.pyplot as plt


X_train=np.array([[.5,1.5],[1,1],[1.5,0.5],[3,0.5],[2,2],[1,2.5]])
y_train=np.array([0,0,0,1,1,1])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_gradient_logistic(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    
    for i in range(m):
        z=np.dot(w,X[i])+b
        f_wb_i=sigmoid(z)
        err_i=f_wb_i-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err_i*X[i,j]
        dj_db=dj_db+err_i
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_db,dj_dw

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )

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

def gradient_descent(X,y,w_in,b_in,alpha,num_iters):
    
    J_history = []
    w=copy.deepcopy(w_in)
    b=b_in
    
    for i in range(num_iters):
        
        dj_db,dj_dw=compute_gradient_logistic(X,y,w,b)
        
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if i<10000:
            J_history.append(compute_cost_logistic(X,y,w,b))
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
        
    return w,b,J_history
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
        
        
        
        

