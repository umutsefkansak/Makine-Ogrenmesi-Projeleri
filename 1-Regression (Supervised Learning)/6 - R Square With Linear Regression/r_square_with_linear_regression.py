# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:54:33 2024

@author: umut
"""

import pandas as pd

df = pd.read_csv("linear_regression_dataset.csv",sep=";")

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

#%%
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

#%%
from sklearn.metrics import r2_score

print("r_square score: ",r2_score(y,y_head))