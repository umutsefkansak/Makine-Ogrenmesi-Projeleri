

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

df = pd.read_csv("heart.csv")

df.info()

print(df.isna().sum())
print("Total null values: ",df.isnull().sum().sum())

#%%

#  0= less chance of heart attack 1= more chance of heart attack

less_chance = df[df.output == 0]
more_chance = df[df.output == 1]

#%%

plt.scatter(less_chance.trtbps,less_chance.chol,color="green",label="Less chance of heart attack",alpha = 0.7)
plt.scatter(more_chance.trtbps,more_chance.chol,color = "red",label="More chance of heart attack",alpha = 0.7)
plt.legend()
plt.show()

#%%

x_data = df.drop(["output"],axis = 1)
y = df.output.values
#%%
# Normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.2,random_state = 42)

#%%

from sklearn.neighbors import KNeighborsClassifier

score_list = []

for i in range(1,15):
    
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test.values,y_test))

plt.plot(range(1,15),score_list)
plt.show()


#%%

best_k = 0
for i in range(len(score_list)):
    if score_list[i] > score_list[best_k]:
        best_k = i


best_k = best_k + 1
print(score_list)
print(best_k)
#%%
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train,y_train)

print("{} nn Score: {} ".format(best_k,knn.score(x_test.values,y_test)))


























