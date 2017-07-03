
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import metrics
import plotly.plotly as py
import plotly.tools as tls
from sklearn.preprocessing import MinMaxScaler
import random
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
np.random.seed(10)
look_back = 29
epoch = 200

np.random.seed(7)
# load the dataset
dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\HSindex.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)

#print (dataframe.head())
Y = dataframe.loc[:,"Close"].values
def create_return(Y):
    Return = []
    for data in range(1,len(Y)):
        Return.append( ((Y[data]-Y[data-1]) / Y[data-1])*100)

    return Return
Return = create_return(Y)

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
X = dataframe.loc[:,"700 HK Equity":"293 HK Equity"]
TrainXR = []
for column in X:

    TrainXR.append(create_return(X[column].values))


TrainX =[]
for Return in TrainXR:
    TrainX.append(create_var(Return,30))

from sklearn.model_selection import StratifiedKFold
# from sklearn.decomposition import PCA, FastICA
# n_comp = 10
# ica = FastICA(n_components=n_comp, random_state=42)
# ica_X = ica.fit_transform(X)
# for i in range(1, n_comp+1):
Train_X = []
for i in range(0,len(TrainX[1])):
    TrainX = np.array(TrainX)
    Train_X.append(TrainX[:,i])

Train_X = np.array(Train_X[:-1])
print(Train_X.shape)

from sklearn.utils import resample

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle = True, random_state = 42)




Train_X = np.reshape(Train_X,(Train_X.shape[0],1,Train_X.shape[1]))
cvscores = []
import time
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
    model.fit(Train_X[train],Var[train], epochs= 2500,batch_size = 32,verbose = 2,shuffle = False)
    scores = model.evaluate(Train_X[test], Var[test], verbose=0)
    cvscores.append(scores*100)


end = time.time()
print("Time: %s" %([end-start]))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
plt.plot( np.divide((Var-model.predict(Train_X).flatten()),Var)*100 )

plt.show()
plt.plot(Var)
plt.plot(model.predict(Train_X))
plt.show()
weight = []
weight.append(1.1)
