


import pandas as pd
import numpy as np
#%%

df = pd.read_csv("heart.csv")


print(df.isna().sum())

#%%

x_data = df.drop(["output"],axis=1)
y = df.output.values

#%%

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3,random_state=42)

#%%

from sklearn.svm import SVC 

svm = SVC()

svm.fit(x_train,y_train)

print("Accuracy : ",svm.score(x_test,y_test))
