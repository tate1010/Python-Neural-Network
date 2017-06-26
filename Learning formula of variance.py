from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import pydot
import graphviz
import tensorflow as tf
from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras import metrics
import random
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
import time
from keras.layers import Dropout
import random
np.random.seed(10)
import math


#generate input
input = []
output = []
for i in range(30,5000):

     data = random.sample(range(100), 30)
     data.sort()
     output.append([np.var(data)])
     mean = np.mean(data)


     input.append(list(data -mean))



print(input[0])
print(output[0])



model = Sequential()
model.add(Dense(30,input_dim= 30, activation = "relu"))
for i in range(15):
    model.add(Dense(64,input_dim= 30, activation = "relu"))

model.add(Dense(1))
model.compile(loss='mse', optimizer = 'adam')

startime= time.time()
history_model = model.fit(input,output,epochs = 5000, batch_size = 256, verbose = 2, validation_split=0.2)
endtime = time.time()
print("time:")
print(endtime-startime)

plt.figure(figsize=(20,10))
plt.plot(np.divide((output-model.predict(input)),output)*100)

test = np.array([random.sample(range(100), 30)])


test = test - np.mean(test)
test.sort()
print(test)

print(model.predict(test))
print(np.var(test))
plt.show()
