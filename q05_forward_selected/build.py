# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here

def forward_selected(df, model):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_list = list(X.columns)
    best_X = []
    best_r2 = []
    
    while len(X_list) > 0:
        r2_X = []
        
        for Xcol in X_list:
            best_X.append(Xcol)
            model.fit(X[best_X], y)
            r2 = model.score(X[best_X], y)          
            r2_X.append((r2, Xcol))
            
            best_X.remove(Xcol)
            
        r2_X.sort()
        score, col = r2_X.pop()
        
        X_list.remove(col)
        
        best_X.append(col)
        best_r2.append(score)
    return best_X, best_r2

# forward_selected(data, model)


