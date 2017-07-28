from IPython.display import clear_output

##lin alg
import numpy as np
import math
import pydot
import graphviz
##Tf
import tensorflow as tf
##data
import matplotlib.pyplot as plt
import pandas
from pandas import Series , DataFrame, Panel
##NN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras import metrics
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.layers import Dropout

##rng
import random

from IPython.display import SVG
#ulti
import time
from scipy.stats import randint as sp_randint
np.random.seed(10)



#generate input
input = []
middle = []
output = []


#generating random data, 5000 in total within the range of 100, 30 numbers each.
for i in range(1,5000):

     data = random.sample(range(100), 30)
     data.sort()
     middle.append(np.square(data))
     output.append([np.std(data)])
     mean = np.mean(data)


     input.append(list(data -mean))


middle = np.array(middle)

#model compile time
startime= time.time()

#machine learning model here
model2 = Sequential()
model2.add(Dense(30,input_dim= 30, activation = "relu"))
model2.add(Dense(246, activation = "relu"))
model2.add(Dense(246, activation = "relu"))

model2.add(Dense(1))
model2.compile(loss='mse', optimizer = 'adam')

#history_model = model2.fit(input,output,epochs = 5000, batch_size = 512, verbose = 2, validation_split=0.2)
########
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor



#random forrest model
clf = RandomForestRegressor(n_estimators=20)

##parameter dictionary
param_dist = {"max_depth": [30,20,10,5,3, None],
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(2, 30),
              "bootstrap": [True, False],
              }
##perform random search with parameter dictionary on random forrest
random_search = RandomizedSearchCV(clf,param_distributions=param_dist, n_iter=100,verbose = 2)

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


start = time.time()
# random_search.fit(input, output)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
      # " parameter settings." % ((time.time() - start), 20))
# report(random_search.cv_results_,100q   )

#


#
from plotly.graph_objs import Scatter, Figure, Layout, Choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go



#perform random forrest on the best model from random search
clf2= RandomForestRegressor(bootstrap= False,max_depth= 3, min_samples_leaf= 17, min_samples_split= 20, n_estimators=100)
clf2.fit(input,output)
endtime = time.time()
print("time:")
print(endtime-startime)
ptt = clf2.predict(input)
output = np.array(output)
print(ptt)
pttt = output.flatten()-ptt
print(pttt)
ptttt = np.divide(pttt,output.flatten())
print(np.absolute(ptttt).mean())


#plotting, graphing result and printing a sample test case.
plt.figure(figsize=(20,10))
test = np.array([random.sample(range(100), 30)])
test = test - np.mean(test)
test.sort()

a = list(range(1,len(ptttt)))
GS = go.Scatter(
        x= a,
        y =(ptttt)
)
fig = dict(data=[GS])
plot(fig)
