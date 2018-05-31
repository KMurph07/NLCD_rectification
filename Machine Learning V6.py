# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:48:07 2018

@author: kmurphy
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import expon
import numpy as np

# Import training data as dataframes

inPath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Clean_Spectral/"
outPath = "N:/Science/GISData/TOS/TOS_Workspace/kmurph/Spectro Data/Output/"

DF_training = pd.read_csv(inPath + "DF_training.csv")
PH_training = pd.read_csv(inPath + "PH_training.csv")
SS_training = pd.read_csv(inPath + "SS_training.csv")
OW_training = pd.read_csv(inPath + "OW_training.csv")
DHI_training = pd.read_csv(inPath + "DHI_training.csv")
EF_training = pd.read_csv(inPath + "EF_training.csv")

# Combine .csvs
nlcd_train = pd.concat([DF_training, PH_training, SS_training, OW_training, DHI_training, EF_training])

# Separate into data vs column name
nlcd_x_train = pd.DataFrame(nlcd_train.loc[:, 'B1':'B426'])
nlcd_y_train = pd.DataFrame(nlcd_train['NLCD'])

# Split into train and test data randomly
x_train, x_test, y_train, y_test = train_test_split(
    nlcd_x_train, nlcd_y_train, test_size = 0.2)

# Create dictionary of parameters to be used in the randomized search function
params = {"PolynomialFeatures_k": [1,2,3],
          "SelectKBest_score_func": ['mutual_info_classif', 'f_classif', 'f_regression'],
          "RandomForestClassifier_max_depth": [3, 5, 7, 10]}

# Make pipeline to test estimators
pipe = make_pipeline(PolynomialFeatures(), SelectKBest(), RandomForestClassifier())

# Fit on pipeline, although causes memory error on laptop
pipe.fit(x_train, y_train)

# Fit on randomized search cv
search = RandomizedSearchCV(pipe, param_distributions = params)
search.fit(x_train, y_train)

## Test from Mueller code
#rf = RandomForestClassifier(n_estimators = 50)
#rf.fit(x_train, y_train)
#print(rf.score(x_test, y_test))
#
#from sklearn.grid_search import GridSearchCV
#search = GridSearchCV(rf, param_grid = {'max_depth': [1,3,5,10]}, cv = 5)
#search.fit(x_train, y_train)
#
## Mueller code -- randomize parameter search
#pipe = Pipeline([("feature_selection", SelectKBest()), ("classifier", SVC())])
#search = RandomizedSearchCV(pipe, params, verbose = 10)
#search.fit(x_train, y_train)
#print(search.best_params_)

