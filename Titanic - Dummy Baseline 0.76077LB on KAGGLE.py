from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras import metrics
from keras.utils import np_utils

import random


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'Pclass':'Embarked'],
                      test.loc[:,'Pclass':'Embarked']))

all_data = all_data.fillna(all_data.mean())

age = all_data.loc[:,'Age']
age.loc[age[1] < 18, age[1]] = 0
print(age)
all_data = all_data.drop(['Cabin','Ticket','Name'],1)


all_data= pd.get_dummies(all_data)


TrainX = all_data[:train.shape[0]]
TestX = all_data[train.shape[0]:]


y = train.Survived
TrainY = np_utils.to_categorical(y)
print(TrainY)

model = Sequential()
model.add(Dense(256,input_dim = TrainX.shape[1],activation = 'relu'))
model.add(Dense(256, activation = "relu"))
model.add(Dense(256))
model.add(Dense(TrainY.shape[1], activation = "softmax"))
model.compile(loss= "binary_crossentropy", optimizer = "RMSprop", metrics=['acc'])
model.fit(TrainX.values, TrainY, epochs= 1000, batch_size = 10 ,verbose = 2)

Prediction = model.predict_classes(TestX.values)

pd.DataFrame({"PassengerId":list(test.loc[:,'PassengerId']), "Survived": Prediction}).to_csv("submission.csv", index=False)


