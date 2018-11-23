# -*- coding: utf-8 -*-
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


from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='once')

df_efixs = pd.read_csv('fixation_data2.csv')
df_saccs = pd.read_csv('saccade_data2.csv')
df_overview = pd.read_csv('overview2.csv')


for feature in ['ampl','dur','pv']:
    clean_feature = f'clean_{feature}'
    df_saccs[clean_feature] = df_saccs[feature]
    df_saccs.loc[df_saccs[feature] >= df_saccs[feature].quantile(0.995),clean_feature] = np.nan

for feature in ['aps','dur']:
    clean_feature = f'clean_{feature}'
    df_efixs[clean_feature] = df_efixs[feature]
    df_efixs.loc[df_efixs[feature] >= df_efixs[feature].quantile(0.995),clean_feature] = np.nan

moment_features = [
    np.nanmean,
    np.nanvar,
    partial(stats.skew,nan_policy='omit'), # was omit, hiermee betere performance
    partial(stats.kurtosis, nan_policy='omit') # was omit
]

df_sacc_features = df_saccs\
    .groupby(by=['PersonID', 'TrialNum'])\
    .agg({'clean_dur' : ['size']+moment_features, 
          'clean_ampl' :moment_features,
          'clean_pv' : moment_features}
    )

df_sacc_features.columns = [f'sacc_{"".join(col)}' for col in df_sacc_features.columns]

df_fix_features = df_efixs\
    .groupby(by=['PersonID', 'TrialNum'])\
    .agg({'clean_dur' : ['size'] + moment_features, 'clean_aps' : moment_features})

df_fix_features.columns = [f'fix_{"".join(col)}' for col in df_fix_features.columns]

df_features = df_sacc_features.join(df_fix_features)

print('########################################################')
print('Done extracting features')
print('########################################################')
 


df_overview = df_overview.drop_duplicates()

df_features_nona = df_features.dropna()
df_all_features \
    = df_features_nona.merge(df_overview,left_on=['PersonID','TrialNum'],right_on=['PersonID','trialnum'],validate='one_to_one') 


def myTrain_test_split(df:pd.DataFrame,df_features,epsilon:float, n_iter:int,random_seed:int=None)\
    ->(pd.Series,pd.Series):
        
    props = np.array(df_features['trialtype'].value_counts(normalize=True))
 
    for kk in range(n_iter):
    # Split train
        train_data, val_test_data = train_test_split(
            df,
            test_size=0.4,
            random_state=random_seed,
            shuffle=True
        )
        
        # Split test & val
        test_data, val_data  = train_test_split(
            val_test_data,
            test_size=0.25,
            shuffle=True,
            random_state=random_seed
        )
        
        make_split = lambda df_meta: {
            'meta': df_meta, 
            'features': df_features[df_features.PersonID.isin(df_meta)],
        }
        
        
        train_data = make_split(train_data)
        test_data = make_split(val_test_data)
        proportions1 = train_data['features']['trialtype'].value_counts(normalize=True)
        proportions2 = test_data['features']['trialtype'].value_counts(normalize=True)
        
        correct_split = True
        
        for i in range(len(proportions1)):
            if abs(proportions1[i] - props[i]) > epsilon:
                correct_split = False

        for i in range(len(proportions2)):
            if abs(proportions1[i] - props[i]) > epsilon:
                correct_split = False
  
        if correct_split:
            print('#####################################################')
            print('found correct train_test split')
            print('#####################################################')
            return train_data, test_data, props
        


input_features = [col for col in df_features if 'fix' in col or 'sacc' in col]


output_feature = 'trialtype'




train_data, test_data, train_test_props = myTrain_test_split(df_overview.reset_index().PersonID.drop_duplicates(),df_all_features, 0.017,2500)


def split_kFold(dataframe:pd.DataFrame, n_splits:int, epsilon:float, n_iter:int): 
    import time
    now = time.time()
    base_props = dataframe['trialtype'].value_counts(normalize=True) # check proportions of input dataframe
    for ll in range(n_iter):
        
        
        dataframe['cat'] = np.random.randint(1,n_splits+1, len(dataframe)) # sample random ints in range 1 - k
        props = np.array(dataframe.groupby(by=['trialtype', 'cat']).size().unstack('cat').apply(lambda df: df/df.sum())) #  proportions of all k splits 
        correct_split = True

        for x in range(props.shape[0]): # check if label balance is within range
            for y in range(props.shape[1]):
                if abs(props[x][y] - base_props[x]) > epsilon:
                    correct_split = False

        if correct_split:
            
            print('#####################################################')
            print('found '+ str(n_splits) + ' splits with a running time of '+  str(round(time.time() - now,2)) + ' seconds and ' + str(ll) + ' iterations')
            print('#####################################################')
            return  dataframe, props
            
dataframe_kFold_split,proportions = split_kFold(train_data['features'],10,0.008,2500)
  
dataframe_kFold_split.to_csv('CV_split.csv')




    
    

