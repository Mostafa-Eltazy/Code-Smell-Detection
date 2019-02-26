#this is for loading the dataset
from sklearn.metrics import confusion_matrix , precision_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Dropout
from keras.models import Sequential
from keras.regularizers import l2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff
import sklearn as sk

dataset = pd.read_csv('csv_result-godclassnew.csv')
dataset = dataset.dropna()


x= dataset.iloc[1:len(dataset),1:61]
y= dataset.iloc[1:len(dataset),62]
print("/////////////////////")
print(y)
print("/////////////////////")
###########################
x= sk.preprocessing.scale(x)
#try diffrent norlizationnnn 
############################
#### shuffling ,data imbalance
x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.10, random_state=0)
print("hnaaaaa")
#A sequintial Model
model = Sequential()
#First hidden layer
model.add(Dense(256,activation="relu",input_dim=60,kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2, noise_shape=None,seed=None))
#seonde hidden layer
model.add(Dense(256,activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2, noise_shape=None,seed=None))
model.add(Dense(256,activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5, noise_shape=None,seed=None))
#output layer
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print("modelllll")
###################
model_output = model.fit(x_train,y_train,epochs=50,batch_size=20,verbose=1,validation_data=(x_test,y_test),)
###################
print('training accuracy : ', np.mean(model_output.history["acc"]))
print('Validation accuracy : ',np.mean(model_output.history["val_acc"]))
print("Train")
y_pred=model.predict(x_test)

print(y_pred)
rounded = [round(x[0]) for x in y_pred]
#y_pred1 = np.array(rounded,dtype='int64')
print(y_test)
print(rounded)
conf_matrix = confusion_matrix(y_test,rounded)
print("The confusion matrix is thisssssssssssssss")
print(conf_matrix)
#precision_score(y_test,rounded)

