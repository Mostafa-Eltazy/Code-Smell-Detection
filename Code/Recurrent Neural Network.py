#this is for loading the dataset
from sklearn.metrics import confusion_matrix , precision_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Dropout
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout
from keras.regularizers import l2
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff
from numpy import array
import sklearn as sk
from keras.layers import TimeDistributed
from numpy.random import seed
from tensorflow import set_random_seed



##################################################################
#The data from the excel sheet needs to be imported in a dataframe
dataset = pd.read_csv('DatasetName.csv')
dataset = dataset.dropna()
# Determine the columns of the Feature in the data set
x= dataset.iloc[0:len(dataset),1:61]
x=pd.DataFrame.to_numpy(x)
#Determine the Label column
y= dataset.iloc[0:len(dataset),62]
print(x)
x= sk.preprocessing.scale(x)
print(y)
x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.10, random_state=0)
############################################################################
# The data needs to be shaped in 3D to be introduced in the Convotutional Layer 
print('xtrain.shape',x_train.shape)
print('xtest.shape',x_test.shape)
x_train = x_train.reshape((627,60,1))
x_test = x_test.reshape((157,60,1))
##############################################################################
#A sequintial Model
model = Sequential()
model.add(LSTM(64,input_shape=(120,1)))
model.add(Activation('relu'))
model.add(Dropout(0.5, noise_shape=None,seed=None))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dropout(0.5, noise_shape=None,seed=None))
model.add(Dense(1))
model.add(Activation("sigmoid"))
optm = keras.optimizers.adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
model.summary()
###############################################################################\
#Model Training 
print("model")
model_output = model.fit(x_train,y_train,epochs=100,batch_size=20,verbose=1,validation_data=(x_test,y_test),)
###################
print("Train")
y_pred=model.predict(x_test)
print(y_pred)
rounded = [round(x[0]) for x in y_pred]
print(y_test)
print(rounded)
conf_matrix = confusion_matrix(y_test,rounded)
print("The confusion matrix is this")
print(conf_matrix)


