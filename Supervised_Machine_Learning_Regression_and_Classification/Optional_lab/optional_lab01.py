import numpy as np
import matplotlib.pyplot as plt


x_train= np.array([1. , 2.])
y_train =np.array([300. ,500.])



print(x_train.shape[0])

m=x_train.shape[0]

plt.scatter(x_train,y_train,marker='x',c='r')
plt.title("Housing Prices")
plt.xlabel('size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.show()

def compute_model_output(x,w,b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range (m):
        f_wb[i]=w*x[i]+b
    return f_wb


tmp_f_wb=compute_model_output(x_train,200,100)

plt.plot(x_train,tmp_f_wb,c='b',label='Our Prediction')



plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()