# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:11:54 2024

@author: umut
"""

import pandas as pd
import matplotlib.pyplot as plt


# Data

df = pd.read_csv("CarPrice_assignment.csv")

df.head()


#%% Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = df["enginesize"].values.reshape(-1,1)
y = df["price"].values.reshape(-1,1)

linear_reg = LinearRegression()

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

linear_reg.fit(x_train,y_train)

y_head = linear_reg.predict(x_train)

plt.scatter(x,y)
plt.plot(x_train,y_head,color = "red")
plt.show()

print("Linear Regression Score: ",linear_reg.score(x_test, y_test))

#%% Polynomial Regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree=2)

x_poly = polynomial_reg.fit_transform(x_train)

linear_reg = LinearRegression()

linear_reg.fit(x_poly,y_train)

x_new = np.arange(min(x_train),max(x_train),0.01).reshape(-1, 1)
x_new_polynomial = polynomial_reg.transform(x_new)
y_head = linear_reg.predict(x_new_polynomial)

plt.scatter(x_train,y_train)
plt.plot(x_new,y_head,color = "red")
plt.show()
print("Polynomial Regression Score",linear_reg.score(polynomial_reg.transform(x_test),y_test))

#%% Multiple Linear Regression

x = df.iloc[:,[9,10,11,12,13,16,18,19,20,21,22,23,24]].values
y = df["price"].values.reshape(-1,1)

x_train_multi,x_test_multi,y_train_multi,y_test_multi = train_test_split(x, y,test_size=0.2,random_state=3)

linear_reg = LinearRegression()

linear_reg.fit(x_train_multi,y_train_multi)

print("Multiple Linear Regression Score : ",linear_reg.score(x_test_multi, y_test_multi))


#%% Decision Tree

from sklearn.tree import DecisionTreeRegressor

decision_tree_reg = DecisionTreeRegressor(random_state = 1)

decision_tree_reg.fit(x_train_multi,y_train_multi)

print("Decision Tree Regression Score",decision_tree_reg.score(x_test_multi, y_test_multi))
#%% Random Forest
from sklearn.ensemble import RandomForestRegressor
 
random_forest_reg = RandomForestRegressor(n_estimators=40,random_state=42)

random_forest_reg.fit(x_train_multi,y_train_multi)
print("Random Forest Regression score: ",random_forest_reg.score(x_test_multi,y_test_multi))