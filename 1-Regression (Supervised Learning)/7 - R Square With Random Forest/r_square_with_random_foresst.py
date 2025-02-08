# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:40:46 2024

@author: umut
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("random+forest+regression+dataset.csv",sep=";", header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)


y_head = rf.predict(x)

#%%

from sklearn.metrics import r2_score

print("r_score: ",r2_score(y,y_head))


