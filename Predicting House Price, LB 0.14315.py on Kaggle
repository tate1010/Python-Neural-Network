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
import seaborn as sns
from scipy.stats import skew
from scipy.stats.stats import pearsonr


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
                      
                      


all_data = pd.get_dummies(all_data)


all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:
TrainX = all_data[:train.shape[0]]
TestX = all_data[train.shape[0]:]



y = train.SalePrice


model = Sequential()
model.add(Dense(288 , input_dim = TrainX.shape[1], activation = 'relu'))
model.add(Dense(288))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = "adam")
model.fit(TrainX.values, y, epochs= 1000, batch_size = 50 ,verbose = 2)


Prediction = model.predict(TestX.values)
print(Prediction)

pd.DataFrame({"Id":list(test.loc[:,'Id']), "SalePrice": Prediction.ravel()}).to_csv("submission.csv", index=False)

