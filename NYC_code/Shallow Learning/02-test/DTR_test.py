#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:47:33 2020

@author: eric
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


samples_day = 96
days_week   = 7
samples_week = samples_day*days_week
months_year = 12

    
##############################################
#############      START     #################
##############################################
#Architecture parameters
batch_size = 672 
n_timesteps = 5
n_features = 1 



espiras = ['145','418','137','169','157','259','217','106','295','318']

for h in range(4):    
    
    val_loss, train_loss, test_loss = list(), list(), list() 
    with open('DTR_trials_t+'+str(h+1)+'.pkl', 'rb') as f:
        pruebas = pickle.load(f)
    
    for k in range(len(espiras)):        
        # load Dataset
        df_train = pd.read_csv('dataset_TRAIN_NYC/'+ espiras[k]+'train_NYC.csv')  
        y_train = df_train['target'].values 
        X_train = StandardScaler().fit_transform(df_train.iloc[:,1:6].values)
        
        # load Dataset
        df_test = pd.read_csv('dataset_TEST_NYC/'+ espiras[k]+'test_NYC.csv') 
        y_test = df_test['target'].values 
        X_test = StandardScaler().fit_transform(df_test.iloc[:,1:6].values)
       
          
        if h==0:
            y_train = y_train
            y_test = y_test
            X_train = X_train
            X_test = X_test
            
        else:            
            y_train = y_train[h:]
            y_test = y_test[h:]
            X_train = X_train[:-h]
            X_test = X_test[:-h]
        
        #Best hyperparam config
        best = pruebas[k].best_trial    
        params = best['misc']['vals']
        
        
        model = DecisionTreeRegressor(min_samples_split=int(np.array(params['split'])),
                                      min_samples_leaf=int(np.array(params['leaf'])),
                                      max_features=int(np.array(params['features'])))
        
        model.fit(X_train, y_train)
        #Model metrics
        pred = model.predict(X_train)
        train_loss.append(r2_score(y_train, pred))
        pred = model.predict(X_test)
        test_loss.append(r2_score(y_test, pred))
    
        val_loss.append(best['result']['loss']*-1)
    
    #Store metrics
    metrics = {}
    metrics['train'] = train_loss
    metrics['val'] = val_loss
    metrics['test'] = test_loss
    with open('DTR_metrics_t+'+str(h+1)+'.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    


    
print('Closing script...')

