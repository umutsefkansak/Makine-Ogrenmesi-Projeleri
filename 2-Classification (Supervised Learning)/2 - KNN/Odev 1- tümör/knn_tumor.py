


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df = pd.read_csv("data.csv")


# dropping useless values
df.drop(["id","Unnamed: 32"],axis = 1, inplace = True)

df.info()

#%%
# Malignant and Benign Tumors
M = df[df.diagnosis == "M"]
B = df[df.diagnosis == "B"]

#%%
# visualization
plt.scatter(M.radius_mean,M.texture_mean,color = "red",label="Melignant tumor",alpha= 0.5)
plt.scatter(B.radius_mean,B.texture_mean,color = "green",label = "Benign tunor",alpha = 0.5)
plt.legend()
plt.show()

#%%

# Converting diagnosis values to numeric
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

#%%

# x and y
x_data = df.drop(["diagnosis"],axis=1)
y = df.diagnosis.values

#%%
#Normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%

# Train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)



#%%# Knn model
from sklearn.neighbors import KNeighborsClassifier


# Finding best k value

score_list = []

for i in range(1,15):
    
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test.values,y_test))


plt.plot(range(1,15),score_list)
plt.xlabel("k neighbor")
plt.ylabel("Accuracy")
plt.show()


#%%
# We can see best value of k in the plot it is 11, let's see in code
 
best_k = 0
for i in range(len(score_list)):
    
    if score_list[i] > score_list[best_k]:
        best_k = i


best_k = best_k + 1 # because score_list[0] k = 1, score_list[1] k = 2,...score_list[n] k = n+1

print(score_list)
print("Best value of neighbor = ",best_k)


#%%
# Knn model
knn = KNeighborsClassifier(n_neighbors=best_k) 
knn.fit(x_train,y_train)

prediction = knn.predict(x_test.values)

print("{} nn score: {} ".format(best_k,knn.score(x_test.values,y_test)))




