from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def map_readmitted_col(data):
    if data == 'NO':
        return 0
    else:
        return 1
        
def map_icd9_codes(data):
    final_data = [1 if code.startswith('250') or code.startswith('249') else 0 for code in data]

    return final_data

def scale_numeric_columns(X):
    # only the first 8 columns are numeric.
    cols_to_scale = X[:, 0:8]
    scaler = StandardScaler()
    X[:, 0:8] = scaler.fit_transform((cols_to_scale))
    
    return X