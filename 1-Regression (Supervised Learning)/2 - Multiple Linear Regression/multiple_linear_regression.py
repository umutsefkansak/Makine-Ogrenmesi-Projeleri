# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:31:47 2024

@author: umut
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)


#%% fitting
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)


print("b0: ",multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)


#predict
multiple_linear_regression.predict([[10,35]])
multiple_linear_regression.predict([[5,35]])

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))
