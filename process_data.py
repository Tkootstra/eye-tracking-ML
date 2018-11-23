# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:07:29 2018

@author: timok
"""



import json
import pandas as pd
from glob import glob
from multiprocessing import Pool, freeze_support
import itertools


DATA_PATH = 'Data_filtered machine learning'

def load_file(file_path:str)->dict:
    with open(file_path,encoding='utf-8') as file:
        return json.load(file)
    

def load_files(data_path:str,cores:int=8)->[dict]:
    
    
        
    file_list = glob(f'{data_path}/*/*.json')
    SOURCE_FILES = file_list
    # initiate processing pool
    read_pool = Pool(cores)
        
    # Return
    return read_pool.map(load_file,SOURCE_FILES)

eye_data = load_files(DATA_PATH,4)

df_data = pd.DataFrame(list(itertools.chain(*eye_data)))\
    .assign( PersonID = lambda data:\
        data\
            .sourcefile.str.extract(r'(?P<exp>[^\\]+)\\(?P<file>[^\\]+).asc$')\
            .apply(lambda df: df.exp + '_' + df.file,axis=1)
    ).drop('sourcefile',axis=1)\
    .assign( PersonID = lambda df: df.PersonID.astype('category'))\
    .set_index(['PersonID','trialnum'])
    
df_overview = df_data[['blocktype','trialtype']]
df_overview.to_csv('overview2.csv')

df_efixs = pd.concat(
    [ pd.DataFrame(frame)\
         .assign(PersonID=personid, TrialNum = trialnum) \
     for ((personid, trialnum), frame) in df_data.Efixs.items()
    ],
    ignore_index=True)

df_efixs.sort_values(by=['PersonID','TrialNum','timestamp'], inplace=True)
df_efixs.to_csv('fixation_data2.csv',index=False)
df_efixs[lambda df: df.dur<= df.dur.quantile(0.99)].dur.hist(bins=2**5)
df_efixs.columns

df_saccs = pd.concat(
    [ pd.DataFrame(frame)\
         .assign(PersonID=personid, TrialNum = trialnum) \
     for ((personid, trialnum), frame) in df_data.Esaccs.items()
    ],
    ignore_index=True)

df_saccs.sort_values(by=['PersonID','TrialNum','timestamp'], inplace=True)
df_saccs.to_csv('saccade_data2.csv',index=False)