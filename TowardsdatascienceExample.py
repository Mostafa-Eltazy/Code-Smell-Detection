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

dataset = pd.read_csv('csv_result-god-class.csv')

#data = arff.loadarff('god-class.arff')
#dataset = pd.DataFrame(data[0])

#xcol= dataset.iloc[1:1,4:65]
#print(xcol)
#print(dataset[xcol])
#dataset[xcol] = dataset[xcol].replace({'?':''}, regex=True)
#dataset=dataset.replace({'?':''}, regex=True)

x= dataset.iloc[1:420,5:65]
y= dataset.iloc[1:420,65]


x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.10, random_state=0)
x_train.shape , y_train.shape , x_test.shape , y_train.shape
print("hnaaaaa")
#A sequintial Model
model = Sequential()
#First hidden layer
model.add(Dense(100,activation="relu",input_dim=60,kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None,seed=None))
#seonde hidden layer
model.add(Dense(100,activation="relu",kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None,seed=None))
#output layer
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print("modelllll")
###################
model_output = model.fit(x_train,y_train,epochs=500,batch_size=20,verbose=1,validation_data=(x_test,y_test),)
###################
print('training accuracy : ', np.mean(model_output.history["acc"]))
print('Validation accuracy : ',np.mean(model_output.history["val_acc"]))
print("Train")
y_pred=model.predict(x_test)
rounded = [round(x[0])for x in y_pred]
y_pred1 = np.array(rounded,dtype='int64')
confusion_matrix(y_test,y_pred1)
precision_score(y_test,y_pred1)

