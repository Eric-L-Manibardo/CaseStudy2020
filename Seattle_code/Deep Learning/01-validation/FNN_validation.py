#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:47:33 2020

@author: eric
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import os
import pickle
from hyperopt import hp, fmin, tpe, Trials

samples_day = 288
days_week   = 7
samples_week = samples_day*days_week
months_year = 12

def config_device(computing_device):
    if 'gpu' in computing_device:
        device_number = computing_device.rsplit(':', 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

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
    #Reorder array    
    features = np.array(val[0])
    for i in range(1,months_year*2,1):
        features = np.append(features,val[i], axis=0)
    return features  

def build_FNN(batch_size, n_timesteps, n_features, h1, h2, h3):
    input1 = keras.layers.Input(shape=(n_timesteps,))
    hidden1 = keras.layers.Dense(h1, activation='relu')(input1)
    hidden2 = keras.layers.Dense(h2, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(h3, activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)
    model = keras.models.Model(inputs=input1, outputs=output)     
    model.compile(loss='mse', optimizer= 'adam')    
    return model

def objective(params):        
    metricR2 = list()
    for i in range(3):
        #Initialize empty model
        model = build_FNN(batch_size, n_timesteps, n_features,
                      params['h1'], params['h2'],params['h3'] )
        #Model train
        model.fit(X_train[i], y_train[i], epochs=epoch, batch_size=batch_size, shuffle=True, verbose =0)
        #Model validation
        pred = model.predict(X_val[i])
        #Save score
        metricR2.append(r2_score(y_val[i], pred)) 
    print('\nR2: ' + str(np.mean(np.array(metricR2))))
   
    return -1.0 * np.mean(np.array(metricR2))

    
##############################################
#############      START     #################
##############################################
    
#Architecture parameters
config_device('gpu:3')
epoch = 150
batch_size = 672 #samples per week
n_timesteps = 5
n_features = 1 # as we use only one traffic reader for input


# Traffic reader IDs
espiras = ['2','74','90','95','120','145','147','230','233','285']

#Loop about 4 studied forecasting horizons t+1,t+2,t+3,t+4
for h in range(4):  
    #Save evaluation trials    
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
        # for LSTM stateful .reshape(samples_day*days_week*24, 5, 1)
        X_train1 = StandardScaler().fit_transform(featuresTRAIN(features, 2, 3))
        X_train2 = StandardScaler().fit_transform(featuresTRAIN(features, 1, 3))
        X_train3 = StandardScaler().fit_transform(featuresTRAIN(features, 1, 2))   
        
        #special t+1 format
        if h==0:
            y_val = [y_val1, y_val2, y_val3]
            y_train = [y_train1, y_train2, y_train3]
            X_val = [X_val1, X_val2, X_val3]
            X_train = [X_train1, X_train2, X_train3]
        #for the rest of forecasting horizons   
        else:            
            y_val = [y_val1[h:], y_val2[h:], y_val3[h:]]
            y_train = [y_train1[h:], y_train2[h:], y_train3[h:]]
            X_val = [X_val1[:-h], X_val2[:-h], X_val3[:-h]]
            X_train = [X_train1[:-h], X_train2[:-h], X_train3[:-h]]
        
        # Parameter Space
        SPACE_FNN = dict([('h1',  hp.quniform('h1',10,100,1)),
                      ('h2',  hp.quniform('h2',10,100,1)),
                      ('h3',  hp.quniform('h3',10,100,1))
        ])
        trials = Trials()            
        #Try to minimize MSE over different configurations
        best = fmin(objective, SPACE_FNN, algo=tpe.suggest, trials =trials, max_evals=30)
        print(best)
        pruebas.append(trials)
    
    with open('FNN_trials_t+'+str(h+1)+'.pkl', 'wb') as f:
        pickle.dump(pruebas, f)
    
print('Closing script...')

