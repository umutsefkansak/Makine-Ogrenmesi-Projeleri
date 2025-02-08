# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:02:03 2024

@author: umut
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("random+forest+regression+dataset.csv",sep=";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% Fitting and prediction
from sklearn.ensemble import RandomForestRegressor
import numpy as np

rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)


print("7.8 seviyesindeki fiyat: ",rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#%% Visualiton 
plt.scatter(x,y)
plt.plot(x_,y_head,color = "red")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()

