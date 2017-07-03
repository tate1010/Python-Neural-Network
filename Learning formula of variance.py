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
middle = []
output = []
for i in range(1,5000):

     data = random.sample(range(100), 30)
     data.sort()
     middle.append(np.square(data))
     output.append([np.var(data)])
     mean = np.mean(data)


     input.append(list(data -mean))



middle = np.array(middle)

#model1.add(Dense(30,input_dim= 30, activation = "relu"))
#for i in range(17):
#    model1.add(Dense(32, activation = "relu"))
#
#model1.add(Dense(30))
#model1.compile(loss='mse', optimizer = 'Rmsprop')

startime= time.time()

#history_model = model1.fit(input,middle,epochs = 5000, batch_size = 512, verbose = 2, validation_split=0.2)
#print(model1.predict(input))
model2 = Sequential()
model2.add(Dense(30,input_dim= 30, activation = "relu"))
model2.add(Dense(246, activation = "relu"))
model2.add(Dense(246, activation = "relu"))

model2.add(Dense(1))
model2.compile(loss='mse', optimizer = 'adam')

history_model = model2.fit(input,output,epochs = 5000, batch_size = 512, verbose = 2, validation_split=0.2)

endtime = time.time()
print("time:")
print(endtime-startime)

plt.figure(figsize=(20,10))
plt.plot(np.divide((output-model2.predict(input)),output)*100)

test = np.array([random.sample(range(100), 30)])


test = test - np.mean(test)
test.sort()

print(test)

print(model2.predict(test))
print(np.var(test))
plt.show()
