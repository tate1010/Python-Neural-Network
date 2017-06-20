import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.utils import np_utils
from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import metrics
import plotly.plotly as py
import plotly.tools as tls


train = pd.read_csv('C:\\Users\\tcheng\\Documents\\input\\train.csv')

TestX = (pd.read_csv('C:\\Users\\tcheng\\Documents\\input\\test.csv').values).astype(float)
labels = train.ix[1:,0].values
TrainX = (train.ix[1:,1:].values).astype(float)
TrainY = np_utils.to_categorical(labels)

TrainX /= 255
TrainX -= np.std(TrainX)


TrainX= TrainX.reshape(TrainX.shape[0],28,28,1)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Dropout(0.20))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(TrainY.shape[1],activation  = 'softmax'))


model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.fit(TrainX,TrainY, epochs = 64, verbose = 2, batch_size =64)




TestX /= 255
TestX -= np.std(TrainX)
TestX= TestX.reshape(TestX.shape[0],28,28,1)
Prediction = model.predict_classes(TestX , verbose = 2)



pd.DataFrame({"ImageId": list(range(1,len(Prediction)+1)), "Label": Prediction}).to_csv("submissionC.csv", index=False, header=True)
