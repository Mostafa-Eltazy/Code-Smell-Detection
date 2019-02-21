import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('god-class -(classfyier 0&1).csv')
x= dataset.iloc[1:420,4:65]
y= dataset.iloc[1:420,65]
print(x.head(5))
print(y.head(20))
x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.3)
x_train.shape , y_train.shape , x_test.shape , y_train.shape
print("hnaaaaa")
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=61))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
print("modelllll")
###################
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
###################
#Fitting the data to the training dataset
#################################################################
classifier.fit(x_train,y_train, batch_size=10, epochs=100)
################################################################
eval_model=classifier.evaluate(x_train, y_train)
eval_model
print("Train")
y_pred=classifier.predict(x_test)
y_pred =(y_pred>0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)

