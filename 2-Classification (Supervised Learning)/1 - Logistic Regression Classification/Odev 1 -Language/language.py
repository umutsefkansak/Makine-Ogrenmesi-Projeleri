# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:28:16 2024

@author: umut
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_l = np.load('X.npy')
Y_l = np.load('Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

#%%
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
#%%

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

#%%
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

#%%
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T

#%%


def initialize(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

#%%

def forward_backward_propagation(w,b,x_train,y_train):
    
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] 
    
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients
    
#%%
def update(w,b,x_train,y_train,learning_rate,number_of_iterations):
    
    for i in range(number_of_iterations):
        
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
    parameters = {"weight":w,"bias":b}
    
    return gradients,parameters



def prediction(w,b,x_test):
    
    y_head = sigmoid(np.dot(w.T, x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,number_of_iterations):
    
    
    dimension = x_train.shape[0]
    w,b = initialize(dimension)
    
    
    gradients,parameters = update(w, b, x_train, y_train, learning_rate, number_of_iterations)
    
    y_prediction = prediction(parameters["weight"], parameters["bias"], x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - y_test)) * 100))
    
#%%

logistic_regression(x_train, x_test, y_train, y_test, 0.01, 150)
    
    
