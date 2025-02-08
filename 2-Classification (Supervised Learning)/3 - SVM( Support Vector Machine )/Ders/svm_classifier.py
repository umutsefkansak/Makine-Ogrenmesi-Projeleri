

#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#%%


df = pd.read_csv("data.csv")

df.info()
print(df.isna().sum())
print("Total null values",df.isnull().sum().sum())

#%%
df.drop(["id","Unnamed: 32"],axis = 1,inplace=True)
print(df.isna().sum())
print("Total null values",df.isnull().sum().sum())


#%%
M = df[df.diagnosis == "M"]
B = df[df.diagnosis == "B"]
#%%

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Malignant",alpha=0.7)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label ="Benign",alpha=0.7)
plt.legend()
plt.show()

#%%

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

#%%

x_data = df.drop(["diagnosis"],axis=1)
y = df.diagnosis.values

#%%

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3,random_state=42)

#%%

from sklearn.svm import SVC
svm = SVC()

svm.fit(x_train,y_train)

print("Accuracy: ",svm.score(x_test,y_test))






















