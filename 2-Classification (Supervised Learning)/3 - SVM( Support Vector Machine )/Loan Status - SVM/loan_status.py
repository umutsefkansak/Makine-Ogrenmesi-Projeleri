# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:48:39 2024

@author: umut
"""

import pandas as pd
import numpy as np
#%%

df = pd.read_csv("loan_data.csv")


print(df.isna().sum())

df.dropna(inplace=True)
#%%

df.drop(["Loan_ID"],axis=1,inplace=True)
#%%

df.Gender = [1 if each == "Male" else 0 for each in df.Gender]
df.Married = [1 if each == "Yes" else 0 for each in df.Married]
df.Education = [1 if each == "Graduate" else 0 for each in df.Education]


df.Self_Employed = [1 if each == "No" else 0 for each in df.Self_Employed]

#%%

df.Property_Area = [0 if each == "Rural" else 1 if each == "Urban" else 2 for each in df.Property_Area]

#%%     
df.Loan_Status = [1 if each == "N" else 0 for each in df.Loan_Status]

df.Dependents = [1 if each == "1" else 0 for each in df.Dependents]

#%%

x = df.drop(["Loan_Status"],axis=1)
y = df.Loan_Status.values

#%%



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3,random_state=42)


#%%

from sklearn.svm import SVC



svm = SVC()

svm.fit(x_train,y_train)

print("Accuracy: ",svm.score(x_test,y_test))

#%%



from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train,y_train)

print("Accuracy: ",lr.score(x_test,y_test))

#%%

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(x_train,y_train)

print("Accuracy: ",knn.score(x_test.values,y_test))





