import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(r"C:\Users\Amr\Documents\A\Machine-Learning-Specialization-Coursera\C1 - Supervised Machine Learning - Regression and Classification\week1\Optional Labs")
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl


x_train= np.array([1.  , 2.])
y_train=np.array([300.0,200.0])

def compute(x,y,w,b):
    m=x.shape[0]
    cost_num=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_num=cost+cost_num
    total_cost=(1/(2*m))*cost_num
    return total_cost

b_fixed = 100 
w_vals=np.linspace(-300,300,500)
cost_vals= [compute(x_train,y_train,w,b_fixed) for w in w_vals]

plt.figure(figsize=(8, 6))
plt.plot(w_vals,cost_vals,c='b')
plt.xlabel('w')
plt.ylabel('cost')
plt.title(f'Cost vs w (with fixed b = {b_fixed})')
plt.grid(True)
plt.show()  

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()