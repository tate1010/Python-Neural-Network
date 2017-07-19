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
varnum = 30


##

dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\^HSI.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)
Date = dataframe.loc[:,"Date"]
ANS = dataframe.loc[:,"Close"]




##def fundction to create return
def create_return(Y):
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


##def function to create rolling window with variance calculated
def rolling_window_withVar(Y,int,k= 1):
    output = []
    for i in range(int,len(Y)):
        temp = []
        for j in range(i-int,i):
            temp.append(Y[j])
        output.append(math.sqrt(k)*np.std(np.array(temp)))
    return output

##def function to create a rolling window
def rolling_window_pre(Y,int,k = 1):
        output = []
        for i in range(int,len(Y)):
            temp = []
            for j in range(i-int,i):
                temp.append(Y[j])
            output.append((np.array(temp)))
        return output




Return_value_Daily = create_return(ANS.values)
Return_value_Monthly = create_return_month(ANS.values)
print(Return_value_Monthly[0:12])
print(Return_value_Daily[0:252])
print(max(Return_value_Daily[0:252]))
print(min(Return_value_Daily[0:252]))
Rolling_Day = rolling_window_withVar(Return_value_Daily,252,252)
Rolling_month  = rolling_window_withVar(Return_value_Monthly,12,12)

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

sparseness = np.sum(lasso.coef_ == 0)/len(Rolling_Day)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######Reverse ICT
Xhat = idct(lasso.coef_)


##Xhat is the result form CS
###


Reconstructed_Rolling_Variance = rolling_window_withVar(Xhat,252,252)
##machine learnign algrotihm here##


from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

##GBRhyper params
params = {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'verbose': 2 ,'random_state' : 10}
clf = GradientBoostingRegressor(**params)
Y = Rolling_Day[0::20]##using every 20 data from the daily input.
online = []
Rolling_month_pre = rolling_window_pre(Return_value_Monthly,12,12)


#online learning from the last 45 to the last one avalible data.
for x in range(-45,-1):

    TrainX = Rolling_month_pre[:x]
    TestX = Rolling_month_pre[x]
    clf.fit(TrainX, Y[:x])
    online.append(clf.predict(TestX)[0])

###
print(Reconstructed_Rolling_Variance)
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
#### using CS as input#####
CS_online = []

## LSTM here
Rolling_Daily_pre = rolling_window_pre(Xhat,252,252)

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
TrainX=  Rolling_Daily_pre[:x]
TrainX = np.array(TrainX)

TrainY = Rolling_Day[:x]
TrainY = np.array(TrainY)
TrainY = np.reshape(TrainY,(TrainY.shape[0],1))

TrainX= np.reshape(TrainX, (TrainX.shape[0],TrainX.shape[1],1))
## train with all data before 800, 010 epochs .
lstm_model = fit_lstm (TrainX , TrainY,  1 ,100, 29)
##training result
ML_online = lstm_model.predict(TrainX, batch_size = 1 )


    #X = np.reshape(X, (X.shape[0],X.shape[1],1))
CS_online = []

#ONLINE training and fitting
for i in range(x,0):
    X = np.array(Rolling_Daily_pre[i])
    print (X)
    X = np.reshape(X,(1,252,1))
    Y = np.array(Rolling_Day[i])
    Y = np.reshape(Y,(1,1))
    Pred = Rolling_Daily_pre[i+1]
    Pred = np.array(Pred)
    Pred = np.reshape(Pred,(1,252,1))
    for j in range(10):
        print('Generation %d' % (i))

        lstm_model.fit(X, Y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        lstm_model.reset_states()
    Pred = lstm_model.predict(Pred).flatten()
    CS_online.append(Pred)

######plotting

RRM_Plot = go.Scatter(
          x=Date[0::20],
          y=Return_value_Monthly,
          mode = 'lines+markers',
          name = "Monthly Return"

)
RR_Plot = go.Scatter(
          x=Date,
          y=Return_value_Daily,
          name = "Daily Return"
)
Re_Plot = go.Scatter(
            x= Date,
            y= Xhat,
            name = "Reconstructed Daily Return"
)


print(CS_online)
predict_plot_tset = go.Scatter(
            x=Date[240+(len(TrainX)-43)*20::20],
            y=online,
            mode = 'lines+markers',
            name = "GBM on monthly, -Testing, daily updated"
)



Rolling_Day_Plot = go.Scatter(
          x=Date[252:],
          y=Rolling_Day,
          name= "Rolling Daily Annual Return")
Rolling_Month_Plot = go.Scatter(
          x=Date[240::20],
          y=Rolling_month,
          mode = 'lines+markers',
          name = "Rolling Monthly Annual Return")
Rolling_ReDay_Plot = go.Scatter(
          x=Date[252:],
          y=Reconstructed_Rolling_Variance,
          name = "Reconstructed Rolling Daily Annual Return from Monthly")
predict_plot_day = go.Scatter(
          x=Date[-800:],
          y=np.array(CS_online).flatten(),
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

fig = dict(data=[RR_Plot,Rolling_Day_Plot,Re_Plot,Rolling_ReDay_Plot,predict_plot_day,RRM_Plot,Rolling_Month_Plot ],layout = layout)
plot(fig)
