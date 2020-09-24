

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
from hyperopt import hp, fmin, tpe, Trials
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

samples_day = 288
days_week   = 7
samples_week = samples_day*days_week
months_year = 12

def targetVAL(y, semana):
    val = list()
    for i in range(months_year):
         val.append(y[samples_week*((3*i)+semana-1):samples_week*((3*i)+semana-1) + samples_week])                 
    return np.array(val).flatten()
def featuresVAL(y, semana):
    val = list()
    for i in range(months_year):
        val.append(y[samples_week*((3*i)+semana-1):samples_week*((3*i)+semana-1) + samples_week])         
    features = np.array(val[0])
    for i in range(1,months_year,1):
        features = np.append(features,val[i], axis=0)
    return features
def targetTRAIN(y, semana1, semana2):    
    val = list()
    for i in range(months_year):
         val.append(y[samples_week*((3*i)+semana1-1):samples_week*((3*i)+semana1-1) + samples_week])
         val.append(y[samples_week*((3*i)+semana2-1):samples_week*((3*i)+semana2-1) + samples_week])          
    return np.array(val).flatten()
def featuresTRAIN(y, semana1, semana2):
    val = list()
    for i in range(months_year):
        val.append(y[samples_week*((3*i)+semana1-1):samples_week*((3*i)+semana1-1) + samples_week])
        val.append(y[samples_week*((3*i)+semana2-1):samples_week*((3*i)+semana2-1) + samples_week])          
    features = np.array(val[0])
    for i in range(1,months_year*2,1):
        features = np.append(features,val[i], axis=0)
    return features  






def objective(params):        

    metricR2 = list()
    for i in range(3):
        #initialize
        model = XGBRegressor(objective ='reg:squarederror',
            n_estimators=int(params['estimators']),
                                      min_samples_leaf=int(params['leaf']),
                                      min_samples_split=int(params['split']),
                                      max_features=int(params['features']))
        #model train
        model.fit(X_train[i], y_train[i])
        #Model validation
        pred = model.predict(X_val[i])
        
        metricR2.append(r2_score(y_val[i], pred))
    
       
    print('\nR2 medio: ' + str(np.mean(np.array(metricR2))))
    
    return -1.0 * np.mean(np.array(metricR2))

    
##############################################
#############      START     #################
##############################################
    
#Architecture parameters

epoch = 150
batch_size = 672 
n_timesteps = 5
n_features = 1 


# Traffic reader IDs
espiras = ['2','74','90','95','120','145','147','230','233','285']

for h in range(4):        
    pruebas = list() 
    for k in range(len(espiras)):        
        # load Dataset
        df = pd.read_csv('dataset_TRAIN_Seattle/'+ espiras[k]+'train_Seattle.csv')          
        y = df['target'].values 
        features = df.iloc[:,1:6].values
        #Validation data (one week of the first 3 weeks of each month)
        y_val1 = targetVAL(y, 1).reshape(-1,1).squeeze()
        y_val2 = targetVAL(y, 2).reshape(-1,1).squeeze()
        y_val3 = targetVAL(y, 3).reshape(-1,1).squeeze()
        X_val1 = StandardScaler().fit_transform(featuresVAL(features, 1))
        X_val2 = StandardScaler().fit_transform(featuresVAL(features, 2))
        X_val3 = StandardScaler().fit_transform(featuresVAL(features, 3))
        #Training data (remaining two weeks of the first 3 weeks of each month) 
        y_train1 = targetTRAIN(y, 2,3).reshape(-1,1).squeeze()
        y_train2 = targetTRAIN(y, 1,3).reshape(-1,1).squeeze()
        y_train3 = targetTRAIN(y, 1,2).reshape(-1,1).squeeze()
        X_train1 = StandardScaler().fit_transform(featuresTRAIN(features, 2, 3))
        X_train2 = StandardScaler().fit_transform(featuresTRAIN(features, 1, 3))
        X_train3 = StandardScaler().fit_transform(featuresTRAIN(features, 1, 2)) 
        
        if h==0:
            y_val = [y_val1, y_val2, y_val3]
            y_train = [y_train1, y_train2, y_train3]
            X_val = [X_val1, X_val2, X_val3]
            X_train = [X_train1, X_train2, X_train3]
            
        else:            
            y_val = [y_val1[h:], y_val2[h:], y_val3[h:]]
            y_train = [y_train1[h:], y_train2[h:], y_train3[h:]]
            X_val = [X_val1[:-h], X_val2[:-h], X_val3[:-h]]
            X_train = [X_train1[:-h], X_train2[:-h], X_train3[:-h]]
        
        
           # Parameter Space
        SPACE_FNN = dict([('split',  hp.quniform('split',2,100,1)),
                          ('leaf',  hp.quniform('leaf',2,20,1)),
                          ('features',  hp.quniform('features',1,5,1)), 
                          ('estimators', hp.choice('estimators', [50,100,150,300]))
                      
            ])
        trials = Trials()
            
        #Try to minimize MSE over different configurations
        best = fmin(objective, SPACE_FNN, algo=tpe.suggest, trials =trials, max_evals=50)
        print(best)
        pruebas.append(trials)
    
    with open('XGBR_trials_t+'+str(h+1)+'.pkl', 'wb') as f:
        pickle.dump(pruebas, f)
    
print('Closing script...')


