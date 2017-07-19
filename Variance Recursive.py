import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import pandas as pd
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



df = pd.Series([1,2,3,4,5,6,7])

Var = df.ewm(com = 0.5)

print(Var.var())
