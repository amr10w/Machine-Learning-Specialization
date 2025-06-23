import math, copy
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(r"C:\Users\Amr\Documents\A\Machine-Learning-Specialization-Coursera\C1 - Supervised Machine Learning - Regression and Classification\week1\Optional Labs")
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value  

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost =cost + (f_wb-y[i])**2
        
    total_cost=(1/(2*m))*cost
    return total_cost

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    
    for i in range(m):
        f_wb = w * x[i] + b 
        dj_dw=dj_dw + (f_wb-y[i])*x[i]
        dj_db=dj_db + (f_wb-y[i])
   
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
    

def gradient_descent(x,y,w_int,b_int,alpha,num_iters,cost_function,gradient_function):
    j_history=[]
    p_history=[]
    
    b=b_int
    w=w_int
    
    for i in range(num_iters):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        
        b=b-alpha *dj_db
        w=w-alpha * dj_dw
        
        if i<100000:
            j_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
            
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
            f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
            f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w,b,j_history,p_history

w_init=0
b_init=0

iterations=1000
tmp_alpha=1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")    

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)