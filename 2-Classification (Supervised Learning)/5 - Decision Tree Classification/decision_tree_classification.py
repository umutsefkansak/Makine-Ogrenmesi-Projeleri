
import pandas as pd 
import numpy as np
#%%





df = pd.read_csv("data.csv")

df.isna().sum()

print("Total null values: ",df.isnull().sum().sum())

#%%

df.drop(["id","Unnamed: 32"],axis = 1, inplace = True)


#%%

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

#%%


x_data = df.drop(["diagnosis"],axis=1)
y = df.diagnosis.values
#%%

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

#%%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train,y_train)

print("Score: ",dt.score(x_test,y_test))
