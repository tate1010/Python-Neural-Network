#LINE ALG
import numpy as np
import pandas
import math
from pandas import Series , DataFrame, Panel
#preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#plot
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
#range
import random
##nn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import metrics
from keras.callbacks import EarlyStopping
##
np.random.seed(10)
look_back = 29
epoch = 200

np.random.seed(7)
# load the dataset
dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\HSindex.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)

#print (dataframe.head())
Y = dataframe.loc[:,"Close"].values

#creating the return value
def create_return(Y):
    Return = []
    for data in range(1,len(Y)):
        Return.append( ((Y[data]-Y[data-1]) / Y[data-1])*100)

    return Return


Return = create_return(Y)
#creating the rollign variance value
def create_var(data,i):
    var = []

    for data in range(i-1,len(Return)):
        temp=[]
        for j in range(data-i+1,data):
            temp.append(Return[j])

        var.append(np.var(temp))
    return var

Var = create_var(Return,30)

Var = np.array(Var[:-1])

print(len(Var))


##locating the index constituents
X = dataframe.loc[:,"700 HK Equity":"293 HK Equity"]
TrainXR = []

for column in X:
    TrainXR.append(create_return(X[column].values))
##converting each daily index constituents value to their's return valu

TrainX =[]

##coverting each daily return value to ROLLING variance value
for Return in TrainXR:
    TrainX.append(create_var(Return,30))

from sklearn.model_selection import StratifiedKFold ## 5fold CV

Train_X = []



#Taking the Transpose of TrainX
for i in range(0,len(TrainX[1])):
    TrainX = np.array(TrainX)
    Train_X.append(TrainX[:,i])

Train_X = np.array(Train_X[:-1])
print(Train_X.shape)

from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

##defining our 5FOLD CV
kf = KFold(n_splits=5,shuffle = True, random_state = 42)



#Reshape for LSTM fitting
Train_X = np.reshape(Train_X,(Train_X.shape[0],1,Train_X.shape[1]))
cvscores = []
import time

#time tracking
start = time.time()
for train,test in kf.split(Train_X):

    model = Sequential()
    model.add(LSTM(256,input_shape = (1,50),activation = 'linear'))
    for i in range(1,2):
            model.add(Dense(256,activation='linear'))
    for i in range(1,3):
            model.add(Dense(256,activation='relu'))
    model.add(Dense(1))
    model.compile(loss="MSE", optimizer = 'Adam')


    callbacks = EarlyStopping(monitor="val_loss",patience=1000)

    #early stopping monitor
    model.fit(Train_X[train],Var[train], epochs= 2500,batch_size = 32,verbose = 2,shuffle = False)
    scores = model.evaluate(Train_X[test], Var[test], verbose=0)
    cvscores.append(scores*100)


end = time.time()
print("Time: %s" %([end-start]))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
plt.plot( np.divide((Var-model.predict(Train_X).flatten()),Var)*100 )


#plotting
plt.show()
plt.plot(Var)
plt.plot(model.predict(Train_X))
plt.show()
