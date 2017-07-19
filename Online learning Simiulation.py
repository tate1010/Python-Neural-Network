##CSV reading,
import pandas
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
##Processing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
## NN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
##plot
from matplotlib import pyplot as plt
#line alg
from math import sqrt
import numpy as np

look_back = 29
np.random.seed(10)
# frame a sequence as a supervised learning problem


dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\HSI1.csv', usecols=[0], engine='python')
datasets = dataframe.values
dataset = datasets.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
dataset= scaler.fit_transform(dataset)
#transforming the dataset to the range of 0 and 1 to fit an LSTM network

###Creating the dataset
##we want train X to be the 30 days input
#where the corresponding Y to be the up coming days

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

#Here Test are used as UNSEEN data to Simiulate the online processing
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train,test = dataset[0:train_size,:], dataset[train_size-look_back:len(dataset),:]
## we will first split up the dataset into portion , then use the function above to prevent LEAK

print(len(train), len(test))


############

##creating the two dataset, training and testing where testing  = UNSEEN data
trainX, trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)



##in keras LSTM network take a 3D Matrix
trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1], 1))
#testX = np.reshape(testX, (testX.shape[0],1, testX.shape[1]))

#print(trainX)
batch_size  = 1
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, ):
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
    model.compile(loss="MSE", optimizer='adam')

    for i in range(nb_epoch):#A stateful LSTM network must be manually resetted in each state
        print('Generation %d' % (i))
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model

lstm_model = fit_lstm(trainX, 1,1 )
trainPredict = lstm_model.predict(trainX, batch_size = batch_size)
lstm_model.reset_states()

trainPredictplot = np.empty_like(dataset)
trainPredictplot[:] = np.nan
trainPredictplot[30:train_size] = trainPredict[:]
#Plotting the training data se
plt.plot(trainPredictplot)
plt.plot(dataset)
plt.show()


#UPDATEING BEGIN HERE
#########################

#define a method _update_model which refit the network with the new data (similar to the one within the training data)
def update_model(model, X, y,  batch_size, updates):

    #X = np.reshape(X, (X.shape[0],X.shape[1],1))
    for i in range(updates):
        print('Generation %d' % (i))
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model




#while(1==1):

##ONLINE SIMIULATION BEGIN HERE
sim = 0 ##simulation generation 0
for i in test:


    #I is the incoming data
    new_datapoint  = i
    new_datapoint = np.array(list([new_datapoint]))
    # last 30 days =  testX1



    new_x = np.reshape(testX[sim],(1,look_back,1))
    print(new_x)

    ##in here, new_x is the data input on T, and new_datapoint = the answer for T
    new_model = update_model(lstm_model,new_x , new_datapoint, 1,30)

##updateing the model to fit this X and Y pair

#define the value of tmr to be TMR_X
    tmr_x = np.reshape(testX[sim+1],(1,look_back,1))

#make prediction on TMR_X
    prediction = lstm_model.predict(tmr_x, batch_size=batch_size)
    lstm_model.reset_states()


#appending, house cleaning and plotting
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


#reset, next gen
    sim += 1
