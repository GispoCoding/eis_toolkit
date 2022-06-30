from sklearn import preprocessing
import numpy as np

x = np.array([[ 1., -1.,  2.],
             [1., 1., 1.]])
scaler = preprocessing.StandardScaler().fit(x)
ka = scaler.mean_

print(ka)