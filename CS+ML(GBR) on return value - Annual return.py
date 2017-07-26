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


##def function to create rolling window with STDiance calculated
def rolling_window_withSTD(Y,int,k= 1):
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
#creating return value
Return_value_Daily = create_return(ANS.values)
Return_value_Monthly = create_return_month(ANS.values)

print(Return_value_Monthly[0:12])
print(Return_value_Daily[0:252])
print(max(Return_value_Daily[0:252]))
print(min(Return_value_Daily[0:252]))

#creating rolling window STd
Rolling_Day = rolling_window_withSTD(Return_value_Daily,252,252)
Rolling_month  = rolling_window_withSTD(Return_value_Monthly,12,12)

########CS
from scipy.fftpack import dct, idct, fft, ifft ,dst,idst
from scipy.sparse import coo_matrix
D = dct(np.eye(len(Return_value_Daily)))

##take every 20 value for the idenity matrix..
A = D[20::20]
np.shape(A)
from sklearn.linear_model import Lasso
#LASSO
lasso = Lasso(alpha=1.5,max_iter=200000, fit_intercept = True, precompute  = True)
lasso.fit(A,Return_value_Monthly)

# plt.plot(lasso.coef_)

sparseness = np.sum(lasso.coef_ == 0)/len(Rolling_Day)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######Reverse ICT
Xhat = idct(lasso.coef_)
##Xhat is the result form CS
Reconstructed_Rolling_STDiance = rolling_window_withSTD(Xhat,252,252)
##machine learnign algrotihm here##


from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

##GBRhyper params
params = {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'verbose': 2 ,'random_state' : 10}
clf = GradientBoostingRegressor(**params)
Y = Rolling_Day[0::20]##using every 20 data from the daily input.
Rolling_month_pre = rolling_window_pre(Return_value_Monthly,12,12)


print(Reconstructed_Rolling_STDiance)
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
#### using CS as input#####
Testing_Result = []

Reconstructed_Rolling_STDiance = np.array(Reconstructed_Rolling_STDiance)

Rolling_Daily_pre = rolling_window_pre(Xhat,252,252)
x = -30


#define X and Y valueable for training
TrainX=  Xhat[:x]
TrainX = np.array(TrainX)
TrainX = np.reshape(TrainX,(TrainX.shape[0],1))
TrainY = Return_value_Daily[:x]
TrainY = np.array(TrainY)

from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
clf = GradientBoostingRegressor(n_estimators = 10000, verbose = 2)


#random search to find the hyperparameter
#param to search
param_dist = {"max_depth": [30,20,10,5,3, None],
              "learning_rate": [1,0.1,0.01,0.001],
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(2, 30),

              }

###function to report the result
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
import time
start = time.time()

#perform random search algrotihm


# random_search = RandomizedSearchCV(clf,param_distributions=param_dist, n_iter=50,verbose = 2)
# random_search.fit(TrainX, TrainY)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
       # " parameter settings." % ((time.time() - start), 20))
# report(random_search.cv_results_,5  )

#
##Actual GBR here
clf = GradientBoostingRegressor(n_estimators = 100000, min_samples_leaf = 12, min_samples_split = 4, max_depth= None, learning_rate = 0.01, verbose = 2)
#fitting the datas
clf.fit(TrainX,TrainY)
##training result
Training_result = clf.predict(TrainX)

#
#
TestX = np.reshape(Xhat[x:],(Xhat[x:].shape[0],1))
Testing_Result = clf.predict(TestX)

#
#
Appended_result_MLCS = np.append(Training_result,Testing_Result)

#
#
Rolling_annual_for_predicted_Return_values = rolling_window_withSTD(Appended_result_MLCS,252,252)

#
#
######plotting

#
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

#
#
print(Testing_Result)
predict_plot_tset = go.Scatter(
            x=Date[:len(Training_result)],
            y=Training_result,
            mode = 'lines+markers',
            name = "CS+ML, return value- training"
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
          y=Reconstructed_Rolling_STDiance,
          name = "Reconstructed Rolling Daily Annual Return from Monthly")
predict_plot_day = go.Scatter(
          x=Date[x-1:],
          y=Testing_Result,
          name = "CS+ML- return value, testing"
)
#
Rolling_annual_for_predicted_Return_values_plot = go.Scatter(
        x=Date[252:],
        y=Rolling_annual_for_predicted_Return_values,
        name = "CS+ML,Rolling Annual return"
)

#
#

#
#
#
layout = dict(
    title = "Rolling Annual Return ",

#
#
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

#
#
fig = dict(data=[RR_Plot,Rolling_Day_Plot,Re_Plot,Rolling_ReDay_Plot,predict_plot_day,RRM_Plot,Rolling_Month_Plot ,predict_plot_tset,Rolling_annual_for_predicted_Return_values_plot],layout = layout)
plot(fig)
