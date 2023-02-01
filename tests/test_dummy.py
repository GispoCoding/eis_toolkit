from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


myML = RandomForestRegressor(
                n_estimators=50, 
                max_features=2, 
                oob_score=True, 
                max_depth=4
)

t = [1,2]
print(t.__len__())