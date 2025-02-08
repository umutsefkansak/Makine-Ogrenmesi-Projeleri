

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
data = pd.read_csv("data.csv")

#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()
#Malignant = M -> kötü huylu tümör
#Benign = B -> iyi huylu tümör

#%%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


#%%
#Scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label = "kötü huylu",alpha=0.5)
plt.scatter(B.radius_mean,B.texture_mean,color = "green",label="iyi huylu",alpha=0.5)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

#%%

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values

#%%


# normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



#%%

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)



#%%
# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 11) # n_neighbors = k
knn.fit(x_train,y_train)
knn.predict(x_test.values)


#%%
print("{} nn score: {} ".format(11,knn.score(x_test.values,y_test)))


#%%

# find  k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test.values,y_test))

plt.plot(range(1,15),score_list)
plt.xlabel("key value")
plt.ylabel("Accuary")


#%%
best_k = score_list.index(max(score_list)) + 1
print(best_k)

