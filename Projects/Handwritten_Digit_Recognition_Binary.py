import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import load_data,sigmoid

X,y= load_data()


model=Sequential(
    [
        tf.keras.Input(shape=(400,)),    #specify input size
        Dense(units=25, activation = "sigmoid"),
        Dense(units=15, activation =  "sigmoid"),
        Dense(units=1 , activation = "sigmoid" )
    ], name="my_model"
)

print("\n\n\n\n\n")
print(model.layers)
print(model.summary())


model.compile(
    
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
)

model.fit(X,y,epochs=20)

prediction = model.predict(X[0].reshape(1,400))

prediction2=model.predict(X[500].reshape(1,400))

print(prediction)
print(prediction2)

if prediction>0.5:
    print("\n\n1")
else:
    print("\n\n0")
if prediction2>0.5:
    print("\n\n1")
else:
    print("\n\n0")

