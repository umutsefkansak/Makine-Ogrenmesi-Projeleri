
# Libraries
import pandas as pd
import numpy as np

#%%

# Data
df = pd.read_csv("data.csv")

print(df.isna().sum())

print("Total null values: ",df.isnull().sum().sum())

#%%

# Dropping useless values
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

#%%

# converting diagnosis values to numeric
df.diagnosis = [1 if each == "M" else  0 for each in df.diagnosis]

#diagnosimap = {"M":1,"B":0}
#df.diagnosis = df.diagnosis.map(diagnosimap)

#%%

# x and y
x_data = df.drop(["diagnosis"],axis=1)
y = df.diagnosis.values
#%%
# Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
#%%

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)

#%%

# Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=20)

rf.fit(x_train,y_train)

print("Random Forest Score: ",rf.score(x_test, y_test))





