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
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
##
#NUMBER OF SPACE
num = 6
varnum = 30


##

dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\^HSI.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)

#print (dataframe.head())
Y = dataframe.loc[:,"Close"].values
def create_return(Y):
    Return = []
    for data in range(1,len(Y)):
        Return.append( ((Y[data]-Y[data-1]) / Y[data-1])*100)

    return Return
Return = create_return(Y)
def create_var(Return,i):
    var = []

    for data in range(i-1,len(Return)):
        temp=[]
        for j in range(data-i+1,data):
            temp.append(Return[j])

        var.append([np.var(temp)])
    return var

##########
n = len(Return)
y = Return
yt = spfft.dct(y, norm='ortho')

# plt.plot(y)
# plt.title('Original (Daily Return Value)')
# plt.show()

#

#
# plt.plot(yt)
# plt.title("DCT of oroginal data")

varO = create_var(y,varnum)
# plt.plot(varO)
# plt.title("variance, original")

########
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
D = dct(np.eye(len(Return)))
A = D[0::num]
np.shape(A)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01,max_iter=10000)
lasso.fit(A,Return[0::num])

# plt.plot(lasso.coef_)

sparseness = np.sum(lasso.coef_ == 0)/len(Return)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######
Xhat = idct(lasso.coef_)
# plt.figure()
# plt.plot(Xhat)
# plt.title('Reconstructed signal')
# plt.show()


def lag(input,lag= 30):
    out = []
    for i in range(lag-1, len(input)):
        temp = []
        for j in range(i,i+lag):
            temp.append(input[i])
        out.append(temp)
    return out

TrainX = lag(Xhat)
TrainX = np.array(TrainX)
varO = np.array(varO)
callbacks=EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='auto')
model = Sequential()
model.add(Dense(128,input_dim = TrainX.shape[1]))
for i in range(5):
    model.add(Dense(168,activation= "relu"))

model.add(Dense(1))
model.compile(loss = 'mse', optimizer='adam')
model.fit(TrainX,varO,epochs=1000,batch_size = 64 , verbose = 2)



predict = model.predict(TrainX)
plt.plot(predict)
plt.show()

print(varO)
plt.plot(predict-varO)
plt.show()
