ure # -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:07:29 2018

@author: timok
"""

import json
import pandas as pd
import numpy as np
import os.path, os
import matplotlib.pyplot as plt
from glob import glob
from multiprocessing import Pool
import itertools
import random
import seaborn as sns
from scipy import interp
import warnings
from scipy import stats
from functools import partial
from statistics import mean

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler





from sklearn.ensemble import RandomForestClassifier
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

# hier een nieuwe split maken als iterable: die kun je dan als CV meegeven in sklearn CV methodes
    

dataframe_kFold_split = pd.read_csv('CV_split.csv')# 1.5% margin

input_features = [x for x in dataframe_kFold_split.columns if 'fix' in x or 'sacc' in x]
output_feature = 'trialtype'

CV_iterator = []


dataframe_kFold_split = dataframe_kFold_split.reset_index()

for x in range(1,11):
    trainIndices = dataframe_kFold_split[dataframe_kFold_split['cat'] != x].index.values.astype(int)
    validationIndices = dataframe_kFold_split[dataframe_kFold_split['cat'] == x].index.values.astype(int)
    CV_iterator.append((trainIndices, validationIndices))



from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
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
         Integer(1,2, name='p')
         
         ,Categorical(['ball_tree', 'kd_tree'], name= 'algorithm'),
         Integer(1,1000, name='n_neighbors')
         
         ]



    
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
#best_params = (""" Best params: \n activation=% \n alpha=%.6f \n hidden_layer_sizes=%d
#      max_iter=% """ % (estimator_gaussian_process.x[0],
#                                                                                                           estimator_gaussian_process.x[1],
#                                                                                                           estimator_gaussian_process.x[2],
#                                                                                                           estimator_gaussian_process.x[3]))
#
#print(best_params)
from skopt import dump

dump(estimator_gaussian_process, 'KNN_Bayesian_results.pkl')

from skopt.plots import plot_convergence
plot_convergence(estimator_gaussian_process)      




























































##TODO: hieronder voor alle classys bouwen - eventueel met bayes search space
#for kkk in range(1):
#    
#    # hier random model samplen
#    
#    # NIEUWE RANDOM SEARCH BOUWEN!!
#    
#    
#    
#    model = RandomForestClassifier(n_estimators=800,
#        criterion='gini',
#        max_features='auto',
#        max_depth=26,
#        min_samples_split=2**6,
#        n_jobs=-1)
#    
#    lent = 0
#    for kk in list(dataframe_kFold_split['cat'].unique()):
#        
#        
#        
#        validation_set = dataframe_kFold_split.loc[dataframe_kFold_split['cat'] == kk]
#    #    validation_set = validation_set[[col for col in validation_set.columns if 'var' in col or 'mean' in col]]
#        lent+= len(validation_set)
#        training_set  = dataframe_kFold_split.loc[dataframe_kFold_split['cat'] != kk]
#    #    training_set = training_set[[col for col in training_set.columns if 'var' in col or 'mean' in col]]
#        
#        #randomCV = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), param_distributions=hyperparams, n_iter=1, cv=[(slice(None), slice(None))])
#        
#        X = training_set[input_features]
#        y = training_set[output_feature]
#            
#        RF_pipeline.fit(X,y)
#        
#        print(evaluate_model(RF_pipeline,training_set,validation_set))
#            
#        y_score = RF_pipeline.predict_proba(validation_set[input_features])
#        y_test = validation_set[output_feature]
#        y_test = np.vstack([y_test==0,y_test==1,y_test==2]).T.astype(float)
#        
#        
#        
#        fpr = dict()
#        tpr = dict()
#        roc_auc = dict()
#        for i in range(3):
#            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#            roc_auc[i] = auc(fpr[i], tpr[i])
#        tprs_0[kk] = tpr[0]
#        tprs_1[kk] = tpr[1]
#        tprs_2[kk] = tpr[2]
#        fprs_0[kk] = fpr[0]
#        fprs_1[kk] = fpr[1]
#        fprs_2[kk] = fpr[2]
#        aucs_0[kk] = roc_auc[0]
#        aucs_1[kk] = roc_auc[1]
#        aucs_2[kk] = roc_auc[2]
#        
#        
#    
##    avg_roc_score.append((np.mean(list(aucs_0.values())) + np.mean(list(aucs_1.values())) + np.mean(list(aucs_2.values()))) /3)
##    min_samples_split.append(2**kkk+1)    
##              
##plt.plot(min_samples_split,avg_roc_score)          
#        
#
#     
#
#               
#print('test perf')
#print(evaluate_pipeline(RF_pipeline, train_data, test_data))
#
#print(len(dataframe_kFold_split) == lent)
#
#plt.figure(figsize=(10,10))
#for i in range(1,len(fprs_0)+1):
#    plt.plot(fprs_0[i], tprs_0[i])
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('10-fold CV ROC for class 0')
#plt.legend(loc="lower right")
#
#textstr = 'mean: '+str(round(np.mean(list(aucs_0.values())),3)) + ' SD: ' + str(round(np.std(list(aucs_0.values())),3))
#plt.text(0.8,0.1,textstr)
#plt.show()
#
#plt.figure(figsize=(10,10))
#for i in range(1,len(fprs_0)+1):
#    plt.plot(fprs_1[i], tprs_1[i])
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('10-fold CV ROC for class 1')
#plt.legend(loc="lower right")
#textstr = 'mean: '+str(round(np.mean(list(aucs_1.values())),3)) + ' SD: ' + str(round(np.std(list(aucs_1.values())),3))
#plt.text(0.8,0.1,textstr)
#plt.show()
#
#plt.figure(figsize=(10,10))
#for i in range(1,len(fprs_0)+1):
#    plt.plot(fprs_2[i], tprs_2[i])
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('10-fold CV ROC for class 2')
#plt.legend(loc="lower right")
#textstr = 'mean: '+str(round(np.mean(list(aucs_2.values())),3)) + ' SD: ' + str(round(np.std(list(aucs_2.values())),3))
#plt.text(0.8,0.1,textstr)
#plt.show()
#
#    
    
    

