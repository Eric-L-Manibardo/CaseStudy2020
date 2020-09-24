#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:24:26 2020

@author: eric
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

espiras = ['145','418','137','169','157','259','217','106','295','318']


for h in range(4):    
    persistencia = list()
    for k in range(len(espiras)):
        
        # load Dataset
        df_train = pd.read_csv('dataset_TRAIN_NYC/'+ espiras[k]+'train_NYC.csv') 
        y_train = df_train['target'].values 
        
        persistencia.append(r2_score(y_train[1+h:], y_train[:-(1+h)]))
    
    print(np.array(persistencia))
