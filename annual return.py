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
########
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
D = dct(np.eye(len(RR)))
A = D[20::20]
np.shape(A)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.99,max_iter=10000000, fit_intercept = True, precompute  = True)
lasso.fit(A,RRM)

# plt.plot(lasso.coef_)

sparseness = np.sum(lasso.coef_ == 0)/len(Rolling_Day)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######
Xhat = idct(lasso.coef_)


Re = rolling_window(Xhat,252,252)

print(Re)
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
fig=plt.figure()
ax=fig.add_subplot(111, label="1")

ax2=fig.add_subplot(111, label="2", frame_on=False)


ax.plot(Rolling_Day, color="C0", label = "Annual Return using Rolling Daily")
ax.plot(Re, color= 'C2', label = "Annual Return using reconstructed Rolling Daily")
ax.set_xlabel("Rolling Daily Annual Return", color="C0")
ax.set_ylabel("Annual Return", color="C0")
ax.tick_params(axis='x', colors="C0")
ax.tick_params(axis='y', colors="C0")
ax.set_ylim([10,50])

ax2.plot(Rolling_month, color="C1", label = "Annual Return using Rolling Monthly")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_ylim([10,50])
ax2.set_xlabel('Rolling Monthly Annual Return', color="C1")
ax2.set_ylabel('Annual Return', color="C1")
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')
ax2.tick_params(axis='x', colors="C1")
ax2.tick_params(axis='y', colors="C1")
ax2.grid(None)

ax.legend()
ax2.legend( borderaxespad=5,loc = 2)
plt.legend()
plt.show()

import plotly.graph_objs as go


Rolling_Day_Plot = go.Scatter(
          x=Date[252:],
          y=Rolling_Day,
          name= "Rolling Daily Annual Return")
Rolling_Month_Plot = go.Scatter(
          x=Date[252::20],
          y=Rolling_month,
          name = "Rolling Monthly Annual Return")
Rolling_ReDay_Plot = go.Scatter(
          x=Date[252:],
          y=Re,
          name = "Reconstructed Rolling Daily Annual Return from Monthly")


layout = dict(
    title = "Rolling Annual Return ",


    legend=dict(  x=0,
        y=1,
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

fig = dict(data=[Rolling_Day_Plot,Rolling_Month_Plot,Rolling_ReDay_Plot],layout = layout)
plot(fig)
