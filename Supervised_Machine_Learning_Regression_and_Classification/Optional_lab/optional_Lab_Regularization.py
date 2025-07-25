import numpy as np
import matplotlib.pyplot as plt
import math,copy


def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    m,n=X.shape
    
    cost=0.
    new_term=0.
    for i in range(m):
        f=np.dot(w,X[i])+b-y[i]
        cost=cost+f**2
    
    for j in range(n):
        new_term=new_term+w[j]**2
    
    cost=cost/(2*m)
    total_cost=cost+((new_term*lambda_)/(2*m))
    return total_cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def sigmoid(z):
    return 1/(1+np.exp(-z))
        
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    m,n =X.shape
    
    cost=0.
    for i in range(m):
        z=np.dot(w,X[i])+b
        f=sigmoid(z)
        cost_i=-y[i]*np.log(f)-(1-y[i])*np.log(1-f)
        cost=cost+cost_i
    cost=cost/(m)
        
    reg_cost=0
    for j in range(n):
        reg_cost+=w[j]**2
    reg_cost=(lambda_/(2*m))*reg_cost
    
    total_cost=cost+reg_cost
    return total_cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def compute_gradient_linear_reg(X, y, w, b, lambda_): 

    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0.
    
    for i in range(m):
        err=np.dot(X[i],w)+b-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err*X[i,j]
        dj_db+=err
    dj_db=dj_db/m
    dj_dw=dj_dw/m
    
    for j in range(n):
        dj_dw[j]=dj_dw[j] +(lambda_/m)*w[j]
    
    return dj_db, dj_dw

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    m,n=X.shape
    
    dj_dw=np.zeros((n,))
    dj_db = 0.0    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                      
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]     
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                 
    dj_db = dj_db/m                                  

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )