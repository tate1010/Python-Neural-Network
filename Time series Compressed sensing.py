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

        var.append(np.var(temp))
    return var

##########
n = len(Return)
y = Return
yt = spfft.dct(y, norm='ortho')

plt.plot(y)
plt.title('Original (Daily Return Value)')
plt.show()


plt.plot(yt)
plt.title("DCT of oroginal data")
plt.show()
varO = create_var(y,30)
plt.plot(varO)
plt.title("variance, original")
plt.show()

from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
D = dct(np.eye(len(Return)))
A = D[0::30]
np.shape(A)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01,max_iter=10000)
lasso.fit(A,Return[0::30])

plt.plot(lasso.coef_)
plt.show()
sparseness = np.sum(lasso.coef_ == 0)/len(Return)
print( "Solution is %{0} sparse".format(100.*sparseness))
###
Xhat = idct(lasso.coef_)
plt.figure()
plt.plot(Xhat)
plt.title('Reconstructed signal')
plt.show()
plt.figure()

plt.plot(Xhat-Return)
plt.title('Error delta')
plt.show()
varN = create_var(Xhat,30)

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
