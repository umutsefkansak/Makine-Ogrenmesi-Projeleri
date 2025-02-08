



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x1 = np.array([8])
y1 = np.array([6])


x2 = np.array([5])
y2 = np.array([5])


x3 = np.array([3])
y3 = np.array([7])



x4 = np.array([15])
y4 = np.array([20])


x5 = np.array([17])

y5 = np.array([22])


x = np.concatenate((x1,x2,x3,x4,x5))
y = np.concatenate((y1,y2,y3,y4,y5))


dictionary = {"x":x,"y":y}
data = pd.DataFrame(dictionary)

plt.scatter(data.x,data.y)
plt.show()


