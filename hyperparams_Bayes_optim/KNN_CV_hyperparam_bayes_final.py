ure # -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:07:29 2018

@author: timok
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.metrics import make_scorer



def customAUC(y_test, preds):
    from sklearn.metrics import roc_curve, auc
    from statistics import mean
    import numpy as np
    y_test = np.vstack([y_test==0,y_test==1,y_test==2]).T.astype(float)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return mean([x for x in roc_auc.values()])

my_auc = make_scorer(customAUC, greater_is_better=True, needs_proba=True)

dataframe_kFold_split = pd.read_csv('CV_split.csv')# 1.5% margin

input_features = [x for x in dataframe_kFold_split.columns if 'fix' in x or 'sacc' in x]
output_feature = 'trialtype'

dataframe_kFold_split = dataframe_kFold_split.reset_index()

def makeCViterable(dataframe): # make iterable of splits for sklearn CV implementation
    CV_iterator = []
    for x in range(1,dataframe['cat'].nunique() +1):
        trainIndices = dataframe[dataframe['cat'] != x].index.values.astype(int)
        validationIndices = dataframe[dataframe['cat'] == x].index.values.astype(int)
        CV_iterator.append((trainIndices, validationIndices))
    return CV_iterator

CV_iterator = makeCViterable(dataframe_kFold_split)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


estimator = KNeighborsClassifier()
pipeline = make_pipeline(StandardScaler(), estimator)

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.searchcv import BayesSearchCV
from sklearn.model_selection import cross_val_score


X = dataframe_kFold_split[input_features]

y = dataframe_kFold_split[output_feature]
binarizer = LabelBinarizer()
y = binarizer.fit_transform(y)



space = [Categorical(['uniform','distance'], name='weights'),
         Integer(5,100, name= 'leaf_size'),
         Integer(1,2, name='p'),
         Categorical(['ball_tree', 'kd_tree'], name= 'algorithm'),
         Integer(1,1000, name='n_neighbors')]



    
@use_named_args(space)
def objective(**params):
    estimator.set_params(**params)
    
    return -np.mean(cross_val_score(pipeline,X,y, cv=CV_iterator, n_jobs=6, scoring='roc_auc'))



from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize


estimator_gaussian_process = gp_minimize(objective, space, n_calls=15, verbose=True)

print("Best score=%.4f" % estimator_gaussian_process.fun)
best_params = list(estimator_gaussian_process.x)

best_model = KNeighborsClassifier(n_neighbors=best_params[4], weights = best_params[0]
, leaf_size=best_params[1], p=best_params[2], algorithm=best_params[3])

print(best_model)

from skopt import dump

dump(estimator_gaussian_process, 'KNN_Bayesian_results.pkl')

from skopt.plots import plot_convergence
plot_convergence(estimator_gaussian_process)      




    

