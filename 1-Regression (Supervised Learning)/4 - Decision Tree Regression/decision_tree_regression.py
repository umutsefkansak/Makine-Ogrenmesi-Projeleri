# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 01:45:09 2024

@author: umut
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision+tree+regression+dataset.csv",sep=";",header=None)


x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
 
 
#%% decision tree regression


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

tree_reg.predict([[6]]) 
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

#%%

plt.scatter(x,y,color = "red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("price")
plt.show()

#%% Kendim ekledim 
# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

y_head2 = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_head2,color="red")
plt.show()

