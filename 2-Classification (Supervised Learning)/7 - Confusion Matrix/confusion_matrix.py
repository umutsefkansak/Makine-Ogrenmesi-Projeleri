

import pandas as pd
import numpy as np

#%%

df = pd.read_csv("data.csv")
print(df.isna().sum())
print("Total null values",df.isnull().sum().sum())

df.head()

#%%

df.drop(["id","Unnamed: 32"],axis= 1,inplace=True)

#%%

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

#%%

x_data = df.drop(["diagnosis"],axis = 1)
y = df.diagnosis.values

#%%
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state = 42)
#%%

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train,y_train)

print("Random Forest Accuracy: ",rf.score(x_test, y_test))


#%%

y_prediction = rf.predict(x_test)
y_true = y_test

#%% Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_prediction)


#%% Confusion Matrix visualization
import seaborn as sns
import matplotlib.pyplot as plt


f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt = ".0f",ax = ax)
plt.xlabel("Y prediction")
plt.ylabel("Y true")
plt.show()

















