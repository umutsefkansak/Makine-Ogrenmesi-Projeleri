# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 01:22:52 2024

@author: umut
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv",sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.show()

#%%
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(x)

#%%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(x_polynomial,y)


#%%
y_head = linear_regression.predict(x_polynomial)

plt.scatter(x,y)
plt.plot(x,y_head,color="red")
plt.show()


