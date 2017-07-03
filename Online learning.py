from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
look_back = 29
np.random.seed(10)
# frame a sequence as a supervised learning problem


dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\HSI1.csv', usecols=[0], engine='python')
datasets = dataframe.values
dataset = datasets.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
dataset= scaler.fit_transform(dataset)

###
def AC_ERROR_RATE(y_true, y_pred):
    where = tf.not_equal(y_true, y_pred)


    return K.sum(tf.where(where))

def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back)]
                mean = np.mean(a)
                # newstd = np.std(np.append(a,dataset[i+look_back+1]))
                #a = a -mean
                dataX.append(a)
                dataY.append(dataset[i+look_back+1])
        return np.array(dataX), np.array(dataY)
# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train,test = dataset[0:train_size,:], dataset[train_size-look_back:len(dataset),:]


print(len(train), len(test))


############
trainX, trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)
#testX, testY = create_dataset(test,look_back)
trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1], 1))
#testX = np.reshape(testX, (testX.shape[0],1, testX.shape[1]))

#print(trainX)
batch_size  = 1
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = trainX, trainY
    model = Sequential()
    model.add(LSTM(29, batch_input_shape=(1, look_back, 1), return_sequences =True ,stateful = True))
    #model.add(BatchNormalization())
    model.add(Dropout (0.2))
    model.add(LSTM(29, batch_input_shape=(1, look_back, 1), return_sequences =False,stateful = True))
    model.add(Dense(32))
    model.add(Dropout (0.2))
    model.add(Dense(32))
    model.add(Dense(output_dim=1))
    model.compile(loss="MSE, optimizer='adam')
    for i in range(nb_epoch):
        print('Generation %d' % (i))
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model

lstm_model = fit_lstm(trainX, 1,1 , 29)
trainPredict = lstm_model.predict(trainX, batch_size = batch_size)
lstm_model.reset_states()


trainPredictplot = np.empty_like(dataset)
trainPredictplot[:] = np.nan
trainPredictplot[30:train_size] = trainPredict[:]
plt.plot(trainPredictplot)
plt.plot(dataset)
plt.show()


#UPDATEING BEGIN HERE
#########################
def update_model(model, X, y,  batch_size, updates):

    #X = np.reshape(X, (X.shape[0],X.shape[1],1))
    for i in range(updates):
        print('Generation %d' % (i))
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model
#while(1==1):
sim = 0
for i in test:
    new_datapoint  = i

    new_datapoint = np.array(list([new_datapoint]))
    # last 30 days =  testX1



    new_x = np.reshape(testX[sim],(1,look_back,1))
    print(new_x)
    new_model = update_model(lstm_model,new_x , new_datapoint, 1,30)

    tmr_x = np.reshape(testX[sim+1],(1,look_back,1))
    prediction = lstm_model.predict(tmr_x, batch_size=batch_size)
    lstm_model.reset_states()
    trainPredict =  np.append(trainPredict,prediction)
    trainPredict = np.reshape(trainPredict, (trainPredict.shape[0],1))
    trainPredictplot = np.empty_like(dataset)
    trainPredictplot[:] = np.nan
    trainPredictplot[30:len(trainPredict)+30] = trainPredict[:]
    print(prediction)
    plt.plot(trainPredictplot)
    plt.plot(dataset)
    testXplot = np.empty_like(dataset)
    testXplot[:] = np.nan
    testXplot[len(trainPredict):len(trainPredict)+29]= testX[sim]
    plt.plot(testXplot)
    plt.show()
    sim += 1
