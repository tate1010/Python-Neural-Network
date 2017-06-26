import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random
from keras import backend as K
import tensorflow as tf
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(42)
#Generate training data
input = []
output = []
realnumber =[]
model_path = "hello.h5"
# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=50, # was 10
        min_delta=0.1,
        verbose=1),

    ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0)
]

for i in range(5000):

     num1 = random.random()*100+1
     num2 = random.random()*100+1
     output.append([math.log(num1)+math.log(num2)])
     realnumber.append([num1*num2])
     input.append([math.log(num1),math.log(num2)])

input = np.array(input)

def RMSE(y_true,y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))


print(input[0:2])
print(output[0:2])


model = Sequential()
model.add(Dense(2,input_dim= 2,activation='relu'))
for i in range(20):
    model.add(Dense(2,input_dim= 2, activation = "linear"))
model.add(Dense(1))


model.compile(loss='MSE',optimizer='adam', verbose=  1 )
model.fit(input,output,epochs = 5000, batch_size = 512 ,verbose = 1 , validation_split=0.2)



plt.plot(np.divide((np.exp(model.predict(input))- realnumber),realnumber)*100)

plt.show()
plt.plot(np.exp(model.predict(input)))
plt.plot(realnumber)
plt.show()


print(np.exp(model.predict(np.array([[math.log(1),math.log(100000)]]))))
