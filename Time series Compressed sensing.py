#LIN ALG
import numpy as np
import cvxpy as cvx
import pandas
import math
##Fourier
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
##NN
from pandas import Series , DataFrame, Panel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import metrics
###Processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
###RNG
import random
###Plotiny
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
##exploring on performing compressive sensing on time series


#
##
#NUMBER OF SPACE
num = 6
varnum = 30


##INPUT

dataframe = pandas.read_csv('C:\\Users\\tcheng\\Documents\\^HSI.csv')
dataframe.dropna(how="any", inplace=True, axis = 0)

#print (dataframe.head())
Y = dataframe.loc[:,"Close"].values

##define function to calculate the return value,

def create_return(Y):
    Return = []
    for data in range(1,len(Y)):
        Return.append( ((Y[data]-Y[data-1]) / Y[data-1])*100)

    return Return
Return = create_return(Y)

##Return_1 = The return from Y1 and Y0..etc


##define function the calculate the rolling variance
def create_var(Return,i):
    var = []

    for data in range(i-1,len(Return)):
        temp=[]
        for j in range(data-i+1,data):
            temp.append(Return[j])

        var.append(np.var(temp))
    return var

##########
n = len(Return)
y = Return
yt = spfft.dct(y, norm='ortho')
##DCT domain of Return vallue



plt.plot(y)
plt.title('Original (Daily Return Value)')
plt.show()
###
plt.plot(yt)
plt.title("DCT of oroginal data")
plt.show()
####
varO = create_var(y,varnum)
plt.plot(varO)
plt.title("variance, original")
plt.show()
########

D = dct(np.eye(len(Return)))
##Np.eye = the idenity matrix of N, where N = len(Return)
##Sampling, We will take every NUM number starting from index 0
A = D[0::num]
##A is the matrix where for each I in column,Row J=  I*20  has a entires of 1 ,else 0

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01,max_iter=10000)

lasso.fit(A,Return[0::num])
##fit

plt.plot(lasso.coef_)
plt.show()
sparseness = np.sum(lasso.coef_ == 0)/len(Return)
print( "Solution is %{0} sparse".format(100.*sparseness))
#######

#IDCT, inverse_transform
Xhat = idct(lasso.coef_)
plt.figure()
plt.plot(Xhat)
plt.title('Reconstructed signal')
plt.show()
plt.figure()


##plotting the  error
plt.plot(Xhat-Return)
plt.title('Error delta')
plt.show()
varN = create_var(Xhat,varnum)

plt.plot(varN)
plt.title("Variance, recovered")
plt.show()
varO=np.array(varO)
varN=np.array(varN)
plt.plot(varO-varN)
plt.title("Error delta for varience")
plt.show()


plt.plot(np.divide(varO-varN,varO)*100)
plt.title("error Precentage for variance")
plt.show()
