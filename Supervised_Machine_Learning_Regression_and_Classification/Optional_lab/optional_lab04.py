import copy, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


x_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train=np.array([460 , 232 , 178]) 

print(f"X shape = {x_train.shape} , dtype= {type(x_train)}")
print(x_train)
print(f"Y shape = {y_train.shape} , dtype= {type(y_train)}")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predict_single_loop(x, w, b): 
    n=x.shape[0]
    p=0
    for i in range(n):
        p_i=x[i] * w[i]
        p=p+p_i
    p=p+b
    return p

x_vec=x_train[0,:]
print(f" x_vec = {x_vec} , and it's shape = {x_vec.shape}")

f_wb=predict_single_loop(x_vec,w_init,b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

def predict_single(x,w,b):
    p=np.dot(x,w)+b
    return p

# get a row from our training data
x_vec = x_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

def compute_cost(X,y,w,b):
    m=X.shape[0]
    cost=0.
    for i in range(m):
        f_wb_i=np.dot(X[i],w)+b
        cost=cost+(f_wb_i-y[i])**2
    cost=cost / (2*m)
    return cost

cost=compute_cost(x_train,y_train,w_init,b_init)

print(f'Cost at optimal w : {cost}')
        
def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
        dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw        
            
    #Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range (num_iters):
        dj_db,dj_dw = gradient_function(X, y, w, b) 
        
        w = w - alpha * dj_dw 
        b = b - alpha * dj_db
        
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

initial_w = np.zeros_like(w_init)
initial_b = 0.

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
      
