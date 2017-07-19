import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
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
import math
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
##
#NUMBER OF SPACE
num = 6
varnum = 30


##

dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\^HSI.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)
Date = dataframe.loc[:,"Date"]
ANS = dataframe.loc[:,"Close"]

def create_return(Y):
    Return = []
    for data in range(1,len(Y)):
        Return.append( ((Y[data]-Y[data-1]) / Y[data-1])*100)

    return Return
def create_return_month(Y):
    J = Y[0::20]
    Return = []
    for data in range(1,len(J)):
        Return.append(((J[data]-J[data-1]) / J[data-1])*100)
    return Return

RR = create_return(ANS.values)
RRM = create_return_month(ANS.values)

print(RRM[0:12])
print(RR[0:252])
print(max(RR[0:252]))
print(min(RR[0:252]))
def rolling_window(Y,int,k= 1):
    output = []
    for i in range(int,len(Y)):
        temp = []
        for j in range(i-int,i):
            temp.append(Y[j])
        output.append(math.sqrt(k)*np.std(np.array(temp)))
    return output
Rolling_Day = rolling_window(RR,252,252)
Rolling_month  = rolling_window(RRM,12,12)


def rolling_window_pre(Y,int,k = 1):
        output = []
        for i in range(int,len(Y)):
            temp = []
            for j in range(i-int,i):
                temp.append(Y[j])
            output.append((np.array(temp)))
        return output



########
from scipy.fftpack import dct, idct, fft, ifft ,dst,idst
from scipy.sparse import coo_matrix
D = dct(np.eye(len(RR)))
A = D[20::20]
np.shape(A)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.5,max_iter=10000, fit_intercept = True, precompute  = True)
lasso.fit(A,RRM)

# plt.plot(lasso.coef_)

sparseness = np.sum(lasso.coef_ == 0)/len(Rolling_Day)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######
Xhat = idct(lasso.coef_)


Re = rolling_window(Xhat,252,252)
##machine learnign algrotihm here##
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'verbose': 2 ,'random_state' : 10}
clf = GradientBoostingRegressor(**params)
Y = Rolling_Day[0::20]


online = []
Rolling_month_pre = rolling_window_pre(RRM,12,12)
for x in range(-45,-1):

    TrainX = Rolling_month_pre[:x]
    TestX = Rolling_month_pre[x]
    clf.fit(TrainX, Y[:x])
    online.append(clf.predict(TestX)[0])

###
print(Re)
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
#### using CS as input#####
CS_online = []


Rolling_Daily_pre = rolling_window_pre(Xhat,252,252)

look_back = 252
batch_size  = 1


# fit an LSTM network to training data
def fit_lstm(TrainX,TrainY, batch_size, nb_epoch, neurons):
    X, y = TrainX, TrainY
    model = Sequential()
    # model.add(LSTM(252, batch_input_shape=(batch_size, look_back, 1), return_sequences =True ,stateful = True))
    #model.add(BatchNormalization())
    # model.add(Dropout (0.2))
    # model.add(LSTM(252, batch_input_shape=(batch_size, look_back, 1), return_sequences =False,stateful = True))


    model.add(Dense(128, input_dim= 1, activation = 'relu'))
    model.add(Dropout (0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout (0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout (0.2))
    model.add(Dense(output_dim=1))
    model.compile(loss="MSE", optimizer='adam')
    for i in range(nb_epoch):
        print('Generation %d' % (i))
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model
x = -200
TrainX=  Re[:-200]
TrainX = np.array(TrainX)
TrainY = Rolling_Day[:-200]
TrainY = np.array(TrainY)
# TrainY = np.reshape(TrainY,(TrainY.shape[0],1))

#
# TrainX= np.reshape(TrainX, (TrainX.shape[0],1))

lstm_model = fit_lstm (TrainX , TrainY,  batch_size ,500, 29)


lstm_model.reset_states()
CS_online = lstm_model.predict(TrainX)
lstm_model.reset_states()
ML_online = lstm_model.predict(np.array(Re[:-200]))



    #X = np.reshape(X, (X.shape[0],X.shape[1],1))
# CS_online = []

#

#
# for i in range(x,0,20):
    # X = np.array(Rolling_Daily_pre[i:i+20])
    # print (X)
    # X = np.reshape(X,(20,252,1))
    # Y = np.array(Rolling_Day[i:i+20])
    # Y = np.reshape(Y,(20,1))
    # Pred = Rolling_Daily_pre[i+1:i+21]
    # Pred = np.array(Pred)
    # Pred = np.reshape(Pred,(20,252,1))
    # for j in range(10):
        # print('Generation %d' % (i))

#
        # lstm_model.fit(X, Y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        # lstm_model.reset_states()
    # Pred = lstm_model.predict(Pred).flatten()
    # CS_online.append(Pred)

#




######

RRM_Plot = go.Scatter(
          x=Date[0::20],
          y=RRM,
          mode = 'lines+markers',
          name = "Monthly Return"

)
RR_Plot = go.Scatter(
          x=Date,
          y=RR,
          name = "Daily Return"
)
Re_Plot = go.Scatter(
            x= Date,
            y= Xhat,
            name = "Reconstructed Daily Return"
)


print(CS_online)
predict_plot_tset = go.Scatter(
            x=Date[240:-200],
            y=ML_online,
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
          y=Re,
          name = "Reconstructed Rolling Daily Annual Return from Monthly")
predict_plot_day = go.Scatter(
          x=Date[240:-200],
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

fig = dict(data=[RR_Plot,Rolling_Day_Plot,Re_Plot,Rolling_ReDay_Plot,predict_plot_day,RRM_Plot,Rolling_Month_Plot,predict_plot_tset ],layout = layout)
plot(fig)