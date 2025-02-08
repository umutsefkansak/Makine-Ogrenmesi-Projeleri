
# kaggle= https://www.kaggle.com/code/umutsefkansak/apple-quality-with-svm-90-and-knn-90/notebook


# Libraries
import pandas as pd
import numpy as np

#%%

# data
df = pd.read_csv("apple_quality.csv")

print(df.isna().sum())
print("Total null values: ",df.isnull().sum().sum())

#%%

# Dropping null values
df.dropna(inplace=True)


#%%

# droppin useless values
df.drop(["A_id"],axis= 1,inplace=True)

# converting object to numeric
df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')
df.Quality = [1 if quality == "good" else 0 for quality in df.Quality]



#%%
# x and y
x_data = df.drop(["Quality"],axis=1)
y = df.Quality.values

#%%
# Normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#%%

# train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

#%%

# support vector machine model
from sklearn.svm import SVC

svm = SVC()

svm.fit(x_train,y_train)

print("SVM Accuracy = ",svm.score(x_test,y_test))

