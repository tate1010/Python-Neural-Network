import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random
from keras import backend as K

#Generate training data
input = []
output = []
for i in range(5000):

     num1 = random.randint(1,1000)
     num2 = random.randint(1,1000)
     output.append([num1+num2])

     input.append([num1,num2])
input = np.array(input)



print(input[0:2])
print(output[0:2])


model = Sequential()
model.add(Dense(128,input_dim= 2))

model.add(Dense(1))


model.compile(loss='mse',optimizer='adam', verbose=  2 ,metrics=['accuracy'])
model.fit(input,output,epochs = 1000, batch_size = 128 , validation_split=0.2)



print(input[0:3])
print(model.predict(input)[0:3])
plt.plot(model.predict(input)- output)
plt.show()



print(model.predict([[1,100000],[2,300000]]))
