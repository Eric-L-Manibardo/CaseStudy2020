#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:24:26 2020

@author: eric
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


espiras = ['4458','6980','10124','6132','3642','4192','3697','3910','3500', '5761']

#Loop about 4 studied forecasting horizons t+1,t+2,t+3,t+4
for h in range(4):    
    persistencia = list()
    for k in range(len(espiras)):        
        # load Dataset
        df_train = pd.read_csv('dataset_TRAIN_MADRID/'+ espiras[k]+'train_MADRID.csv') 
        y_train = df_train['target'].values 
        
        persistencia.append(r2_score(y_train[1+h:], y_train[:-(1+h)]))
    
    print(np.array(persistencia))