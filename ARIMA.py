##lin alg
import numpy as np
import cvxpy as cvx
import random
import math
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
##frame
import pandas
from pandas import Series , DataFrame, Panel
##ploting
import plotly.plotly as py
import plotly.tools as tls
import matplotlib as mpl
import matplotlib.pyplot as plt
#preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
##NN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import metrics
from keras.callbacks import EarlyStopping
#NUMBER OF SPACE
num = 6
STDnum = 30


##

dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\^HSI.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)
Date = dataframe.loc[:,"Date"]
Closing_Value = dataframe.loc[:,"Close"]




##def fundction to create return
def create_return_daily(Y):
    Return = []
    for data in range(1,len(Y)):
        Return.append( ((Y[data]-Y[data-1]) / Y[data-1])*100)

    return Return

##def function to create return for MO
def create_return_month(Y):
    J = Y[0::20]
    Return = []
    for data in range(1,len(J)):
        Return.append(((J[data]-J[data-1]) / J[data-1])*100)
    return Return


##def function to create rolling window with STDiance calculated
def rolling_window_with_std(Y,int,k= 1):
    output = []
    for i in range(int,len(Y)):
        temp = []
        for j in range(i-int,i):
            temp.append(Y[j])
        output.append(math.sqrt(k)*np.std(np.array(temp)))
    return output

##def function to create a rolling window
def rolling_window_without_std(Y,int,k = 1):
        output = []
        for i in range(int,len(Y)):
            temp = []
            for j in range(i-int,i):
                temp.append(Y[j])
            output.append((np.array(temp)))
        return output




Return_value_Daily = create_return_daily(Closing_Value.values)
Return_value_Monthly = create_return_month(Closing_Value.values)
print(Return_value_Monthly[0:12])
print(Return_value_Daily[0:252])
print(max(Return_value_Daily[0:252]))
print(min(Return_value_Daily[0:252]))
Rolling_daily_window_with_std = rolling_window_with_std(Return_value_Daily,252,252)
Rolling_monthly_window_with_std  = rolling_window_with_std(Return_value_Monthly,12,12)

##



########CS
from scipy.fftpack import dct, idct, fft, ifft ,dst,idst
from scipy.sparse import coo_matrix


D = dct(np.eye(len(Return_value_Daily)))

##take every 20 value for the idenity matrix..
A = D[20::20]
np.shape(A)
from sklearn.linear_model import Lasso
#LASSO
lasso = Lasso(alpha=1.5,max_iter=10000, fit_intercept = True, precompute  = True)
lasso.fit(A,Return_value_Monthly)

# plt.plot(lasso.coef_)

sparseness = np.sum(lasso.coef_ == 0)/len(Rolling_daily_window_with_std)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######Reverse ICT
Xhat = idct(lasso.coef_)
##Xhat is the result form CS
###

Reconstructed_Rolling_STDiance = rolling_window_with_std(Xhat,252,252)
#machine learning algo#
Y = Rolling_daily_window_with_std[0::20]##using every 20 data from the daily input.
Rolling_month_pre = rolling_window_without_std(Return_value_Monthly,12,12)


print(Reconstructed_Rolling_STDiance)
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
#### using CS as input#####
Predicted_list = []

## LSTM here
Rolling_reconstructed = rolling_window_without_std(Xhat,252,252)

look_back = 252
batch_size  = 1
# fit an LSTM network to training data
def fit_lstm(TrainX,TrainY, batch_size, nb_epoch, neurons):
    X, y = TrainX, TrainY
    model = Sequential()
    model.add(LSTM(252, batch_input_shape=(1, look_back, 1), return_sequences =True ,stateful = True))
    #model.add(BatchNormalization())
    model.add(Dropout (0.2))
    model.add(LSTM(252, batch_input_shape=(1, look_back, 1), return_sequences =False,stateful = True))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout (0.2))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(output_dim=1))
    model.compile(loss="MSE", optimizer='adam')
    for i in range(nb_epoch):
        print('Generation %d' % (i))
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model

##we wish to use the last 800 data for online prediction
x = -800

##taking everythign until the 800'th from the last
TrainX=  Rolling_reconstructed[:x]
TrainX = np.array(TrainX)
#X is reshaped into (Item, Feature,batch)
TrainX= np.reshape(TrainX, (TrainX.shape[0],TrainX.shape[1],1))




##taking everything until the 800'th from the last
TrainY = Rolling_daily_window_with_std[:x]
TrainY = np.array(TrainY)
##Train Y is reshaped into it transpose,
TrainY = np.reshape(TrainY,(TrainY.shape[0],1))

## train with all data before 800, 010 epochs .
lstm_model = fit_lstm (TrainX , TrainY,  1 ,100, 29)
##training result
ML_online = lstm_model.predict(TrainX, batch_size = 1 )


    #X = np.reshape(X, (X.shape[0],X.shape[1],1))
Predicted_list = []

#ONLINE training and fitting
for i in range(x,0):
    #X is the i'th data from the last
    X = np.array(Rolling_reconstructed[i])
    print (X)
    #here x contain the last 252 daily data
    X = np.reshape(X,(1,252,1))
    #reshape into 1 * 252 * 1. So which is 252 date of indivual data.


    #taking the answer for that date
    Y = np.array(Rolling_daily_window_with_std[i])
    Y = np.reshape(Y,(1,1))

    #define our prediction input to be day t-251 to t+1
    Pred = Rolling_reconstructed[i+1]
    Pred = np.array(Pred)
    Pred = np.reshape(Pred,(1,252,1))

    #refit model using X and Y
    for j in range(10):
        print('Generation %d' % (i))
        #fit the data 10 times
        lstm_model.fit(X, Y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        lstm_model.reset_states()


    #predict T+1
    Pred = lstm_model.predict(Pred).flatten()

    #append prediction to list for plotting
    Predicted_list.append(Pred)

######plotting

monthly_return_plot = go.Scatter(
          x=Date[0::20],
          y=Return_value_Monthly,
          mode = 'lines+markers',
          name = "Monthly Return"

)
Daily_return_plot = go.Scatter(
          x=Date,
          y=Return_value_Daily,
          name = "Daily Return"
)
Reconstructed_Daily_return_plot = go.Scatter(
            x= Date,
            y= Xhat,
            name = "Reconstructed Daily Return"
)


Rolling_daily_annual_return_plot = go.Scatter(
          x=Date[252:],
          y=Rolling_daily_window_with_std,
          name= "Rolling Daily Annual Return")
Rolling_montly_annual_return_plot = go.Scatter(
          x=Date[240::20],
          y=Rolling_monthly_window_with_std,
          mode = 'lines+markers',
          name = "Rolling Monthly Annual Return")
Reconstructed_rolling_daily_annual_return_plot = go.Scatter(
          x=Date[252:],
          y=Reconstructed_Rolling_STDiance,
          name = "Reconstructed Rolling Daily Annual Return from Monthly")
CS_LSTM_plot = go.Scatter(
          x=Date[-800:],
          y=np.array(Predicted_list).flatten(),
          name = "CS+ML, online testing prediction"
)
layout = dict(
    title = "Rolling Annual Return ",


    legend=dict(  x=0,
        y=40,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=12,
            color='#000'
        ),
        bgcolor='#E2E2E2',
        bordercolor='#FFFFFF',
        borderwidth=2
        )
)

fig = dict(data=[Daily_return_plot,Rolling_daily_annual_return_plot,Reconstructed_Daily_return_plot,Reconstructed_rolling_daily_annual_return_plot,CS_LSTM_plot,monthly_return_plot,Rolling_montly_annual_return_plot ],layout = layout)
plot(fig)
