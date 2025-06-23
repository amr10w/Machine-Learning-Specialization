import numpy as np
input_array=np.array([1,2,3])
exp_array=np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)


input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)


def sigmod(z):
    f=1/(1+np.exp(-z))
    
    return f


z_tmp=np.arange(-10,11)
print(z_tmp)

y=sigmod(z_tmp)
np.set_printoptions(precision=3) 
print(y)

print(np.c_[z_tmp,y])