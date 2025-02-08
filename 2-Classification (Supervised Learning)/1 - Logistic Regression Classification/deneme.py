# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:05:50 2024

@author: umut
"""

import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

data.info()
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%% Normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% data split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


#%% initialize w b

def initialize(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

#%%
def sigmoid(z):
    y_head = 1/(1+ np.exp(-z))
    return y_head

#%%

def forward_backward_propagation(w,b,x_train,y_train):
    
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost =  (np.sum(loss))/x_train.shape[1]

    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients

#%%
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    costs = []
    
    for i in range(number_of_iteration):
        
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        costs.append(cost)
        
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
    parameters = {"weight":w,"bias":b}
    
    return costs,parameters    
        
#%%
def predict(w,b,x_test):
    
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
    
    
#%%

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):
    
    dimension = x_train.shape[0]
    w,b = initialize(dimension)
         
    costs,parameters = update(w, b, x_train, y_train, learning_rate, number_of_iteration)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


#%%

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, number_of_iteration = 1000)    
    

#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print(lr.score(x_test.T,y_test.T))
    
    
    
    

