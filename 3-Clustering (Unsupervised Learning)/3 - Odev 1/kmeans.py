


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df = pd.read_csv("column_2C_weka.csv")

df.info()

print(df.isna().sum().sum())

df.isnull().sum()

#%%
import seaborn as sns

sns.pairplot(df,hue="class")


#%%
plt.scatter(df.pelvic_radius,df.degree_spondylolisthesis)
plt.show()

#%%

data = df.loc[:,["pelvic_radius","degree_spondylolisthesis"]]

data2 = df.iloc[:,[4,5]]



#%%

from sklearn.cluster import KMeans

wcss = []

for i in range(1,8):
    
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,8),wcss,'-o')
plt.show()



#%%
kmeans = KMeans(n_clusters=2)

cluster = kmeans.fit_predict(data)

data["label"] = cluster

plt.scatter(data.pelvic_radius[data.label == 0],data.degree_spondylolisthesis[data.label == 0],color="green")

plt.scatter(data.pelvic_radius[data.label == 1],data.degree_spondylolisthesis[data.label == 1],color="red")

#%% Dendogram


from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data2,method="ward")

dendrogram(merg)
plt.show()

#%%


from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")

cluster = hc.fit_predict(data2)

data2["label"] = cluster


plt.scatter(data2.pelvic_radius[data2.label == 0],data2.degree_spondylolisthesis[data2.label == 0],color = "orange")
plt.scatter(data2.pelvic_radius[data2.label == 1],data2.degree_spondylolisthesis[data2.label == 1],color= "purple")
plt.show()

#%% Karşılaştırma

# KMeans
plt.scatter(data.pelvic_radius[data.label == 0],data.degree_spondylolisthesis[data.label == 0],color="green")
plt.scatter(data.pelvic_radius[data.label == 1],data.degree_spondylolisthesis[data.label == 1],color="red")
plt.show()
#%%

# Hierarchical
plt.scatter(data2.pelvic_radius[data2.label == 0],data2.degree_spondylolisthesis[data2.label == 0],color = "orange")
plt.scatter(data2.pelvic_radius[data2.label == 1],data2.degree_spondylolisthesis[data2.label == 1],color= "purple")
plt.show()

#%%
# Orijinal
plt.scatter(df.pelvic_radius[df["class"] == "Abnormal"],df.degree_spondylolisthesis[df["class"]=="Abnormal"],color = "cyan")
plt.scatter(df.pelvic_radius[df["class"]=="Normal"],df.degree_spondylolisthesis[df["class"] == "Normal"],color="purple")
plt.show()
#%%




