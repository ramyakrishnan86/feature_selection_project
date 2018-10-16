# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k = 20):
    X = df.drop(['SalePrice'], axis = 1)
    y = df['SalePrice']

    selector = SelectPercentile(f_regression, k)
    X_new = selector.fit_transform(X, y)
 
    featurelist = list(X.columns.values[np.argsort(selector.scores_)[-1:-X_new.shape[1]-1:-1]])
    
    return featurelist

# percentile_k_features(data,20)


