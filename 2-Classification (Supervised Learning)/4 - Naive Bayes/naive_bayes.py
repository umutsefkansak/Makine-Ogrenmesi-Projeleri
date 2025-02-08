



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df = pd.read_csv("data.csv")

print(df.isna().sum())


#%%
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

#%%

M = df[df.diagnosis == "M"]
B = df[df.diagnosis == "B"]

#%%

plt.scatter(M.radius_mean,M.texture_mean,color="red",label= "Malignant",alpha= 0.7)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Benign",alpha=0.7)
plt.legend()
plt.show()

#%%

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

#%%

x_data =df.drop(["diagnosis"],axis=1)
y = df.diagnosis.values

#%%



x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))


#%%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

#%%

from sklearn.naive_bayes import GaussianNB


nb = GaussianNB()

nb.fit(x_train,y_train)

print("Naive bayes accuracy = ",nb.score(x_test,y_test))



































