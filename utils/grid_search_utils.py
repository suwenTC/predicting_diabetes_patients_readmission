import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_processing_utils import scale_numeric_columns

from operator import itemgetter
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.preprocessing import FunctionTransformer

from sklearn.decomposition import PCA

def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f}, Standard Deviation: {1:.4f}".format(
              score.mean_validation_score,
              score.cv_validation_scores.std()))
        print("Parameters: {0}".format(score.parameters))
        print("")

def evaluate_param(clf, clf_name, parameter, param_values, n_components, index, X_train, y_train):
    pca = PCA(n_components=n_components)
    pipe = Pipeline([
        ('scale_data', FunctionTransformer(scale_numeric_columns)), 
        ('pca', pca),
        (clf_name, clf)
    ])

    # in order to use pipeline with grid search,
    # we have to name the parameter in accordance with convention
    parameter = clf_name+'__'+parameter
    grid_search = GridSearchCV(pipe, cv=3, param_grid={parameter: param_values}, refit=True)
    grid_search.fit(X_train, y_train)

    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(4,2, index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    plt.grid(True)