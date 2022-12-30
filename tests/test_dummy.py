from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


myML = RandomForestRegressor(
                n_estimators=50, 
                max_features=2, 
                oob_score=True, 
                max_depth=4
)


data = [['tom', 10], ['nick', 15], ['juli', 14]]
df2 = pd.DataFrame([[1],[0],[10],[5]], columns=['col'])
v = 7
df1 = pd.DataFrame([[v]], columns=['col'])
dfn = pd.DataFrame([[np.nan]], columns=['col'])
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Name', 'Age'])
dft = 1
