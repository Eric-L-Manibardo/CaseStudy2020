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
from sklearn.ensemble import AdaBoostRegressor

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



espiras = ['4458','6980','10124','6132','3642','4192','3697','3910','3500', '5761']

for h in range(4):    
    
    val_loss, train_loss, test_loss = list(), list(), list() 
    with open('ADA_trials_t+'+str(h+1)+'.pkl', 'rb') as f:
        pruebas = pickle.load(f)
    
    for k in range(len(espiras)):        
        # load Dataset
        df_train = pd.read_csv('dataset_TRAIN_MADRID/'+ espiras[k]+'train_MADRID.csv') 
        y_train = df_train['target'].values 
        X_train = StandardScaler().fit_transform(df_train.iloc[:,1:6].values)
        
        # load Dataset
        df_test = pd.read_csv('dataset_TEST_MADRID/'+ espiras[k]+'test_MADRID.csv') 
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
        estimators=[50,100,150,300]
        loss=['linear','square', 'exponential']
        model = AdaBoostRegressor(n_estimators=estimators[int(np.array(params['estimators']))],
                                  loss=loss[int(np.array(params['loss']))],
                                  learning_rate=np.array(params['lr']))
            
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
    with open('ADA_metrics_t+'+str(h+1)+'.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    


    
print('Closing script...')

