# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:07:29 2018

@author: timok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_curve, auc
from skopt import load
from skopt import space
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def objective(**params):
    estimator.set_params(**params)
    
    return -np.mean(cross_val_score(pipeline,X,y, cv=CV_iterator, n_jobs=4, scoring='roc_auc'))


CV_split = pd.read_csv('CV_split.csv')

base_fpr  = np.linspace(0, 1, 100)


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    
input_features = [x for x in CV_split.columns if 'fix' in x or 'sacc' in x]
output_feature = 'trialtype'

results = load('KNN_Bayesian_results.pkl')
best_params = results.x



    
estimator = KNeighborsClassifier(n_neighbors=best_params[4], weights = best_params[0]
, leaf_size=best_params[1], p=best_params[2], algorithm=best_params[3])

model = make_pipeline(StandardScaler(), estimator)



  

all_tprs = []
tprs_0 = dict()
tprs_1 = dict()
tprs_2 = dict()

fprs_0 = dict()
fprs_1 = dict()
fprs_2 = dict()

aucs_0 = dict()
aucs_1 = dict()
aucs_2 = dict()


plt.figure(figsize=(5, 5))  

for kk in list(CV_split['cat'].unique()):
    
    validation_set = CV_split.loc[CV_split['cat'] == kk]
    training_set  = CV_split.loc[CV_split['cat'] != kk]

    X = training_set[input_features]
    y = training_set[output_feature]
        
    model.fit(X,y)
    
    
        
    y_score = model.predict_proba(validation_set[input_features])
    y_test = validation_set[output_feature]
    y_test = np.vstack([y_test==0,y_test==1,y_test==2]).T.astype(float)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    tprs_0[kk] = tpr[0]
    tprs_1[kk] = tpr[1]
    tprs_2[kk] = tpr[2]
    fprs_0[kk] = fpr[0]
    fprs_1[kk] = fpr[1]
    fprs_2[kk] = fpr[2]
    aucs_0[kk] = roc_auc[0]
    aucs_1[kk] = roc_auc[1]
    aucs_2[kk] = roc_auc[2]
    

plt.figure(figsize=(10,10))
for i in range(1,len(fprs_0)+1):
    plt.plot(fprs_0[i], tprs_0[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.xlim([-0.1, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('10-fold CV ROC for class 0')
plt.legend(loc="lower right")

textstr = 'mean: '+str(round(np.mean(list(aucs_0.values())),3)) + ' SD: ' + str(round(np.std(list(aucs_0.values())),3))
plt.text(0.8,0.1,textstr)
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,len(fprs_0)+1):
    plt.plot(fprs_1[i], tprs_1[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('10-fold CV ROC for class 1')
plt.legend(loc="lower right")
textstr = 'mean: '+str(round(np.mean(list(aucs_1.values())),3)) + ' SD: ' + str(round(np.std(list(aucs_1.values())),3))
plt.text(0.8,0.1,textstr)
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,len(fprs_0)+1):
    plt.plot(fprs_2[i], tprs_2[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.xlim([-0.1, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('10-fold CV ROC for class 2')
plt.legend(loc="lower right")
textstr = 'mean: '+str(round(np.mean(list(aucs_2.values())),3)) + ' SD: ' + str(round(np.std(list(aucs_2.values())),3))
plt.text(0.8,0.1,textstr)
plt.show()
                
        

    


