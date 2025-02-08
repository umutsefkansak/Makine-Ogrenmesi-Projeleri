# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:01:43 2024

@author: umut
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv",sep = ";")


x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)


plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hız")


#Linear Regression -> y = b0 + b1*x
#Multiple Linear Regression -> y = b0 + b1*x1 + b2*x2 +...

#%% Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)


#%%
y_head = lr.predict(x)

plt.scatter(x,y)
plt.plot(x,y_head,color="red")
plt.show()

print("10M tl'lik araba hizi tahmini: ",lr.predict([[10000]]))

#%%
#Polynomial Linear Regression y = b0 + b1*x + b2*x^2 + b3*x^3 + .... + bn*x^n


from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree= 4)


x_polynomial = polynomial_regression.fit_transform(x)

#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)


#%%
y_head2 = linear_regression2.predict(x_polynomial)



plt.scatter(x,y)
plt.plot(x,y_head,color="red")
plt.plot(x,y_head2,color = "black")
plt.show()


linear_regression2.predict([[10000]])


























