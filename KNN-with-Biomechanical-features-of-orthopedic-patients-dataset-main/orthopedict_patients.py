
# kaggle = https://www.kaggle.com/code/umutsefkansak/knn-algorithm-with-90-accuracy/notebook

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%


df = pd.read_csv("column_2C_weka.csv")

df.head()

df.info()

#%%

abnormal = df[df["class"] == "Abnormal"]
normal = df[df["class"] == "Normal"]



#%%

plt.scatter(abnormal.pelvic_incidence,abnormal.pelvic_radius,color = "red",label ="Abnormal",alpha = 0.7)
plt.scatter(normal.pelvic_incidence,normal.pelvic_radius,color="green",label = "Normal",alpha=0.7)
plt.xlabel("Pelvic incidence")
plt.ylabel("pelvic_radius")
plt.legend()
plt.show()

#%%
#Converting class values to numeric
#Nomal -> 0
#Abormal ->1
df["class"] = [1 if each == "Abnormal" else 0 for each in df["class"]]

#%%
# x and y
x = df.drop(["class"],axis=1)
y = df["class"].values


#%%

#Train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

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
# we can see best k value int he plot, let's see it in code

best_k = 0

for i in range(len(score_list)):
    
    if score_list[i] > score_list[best_k]:
        best_k = i

best_k = best_k + 1 # because score_list[0] k = 1, score_list[1] k=2,...,score_lit[n] k = n+1

#%%
# Model

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(x_train,y_train)

print("{} nn Score: {} ".format(best_k,knn.score(x_test.values,y_test)))





