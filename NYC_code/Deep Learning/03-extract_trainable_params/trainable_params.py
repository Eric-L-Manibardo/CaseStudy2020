#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:47:33 2020

@author: eric
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import pickle


samples_day = 271
days_week   = 7
samples_week = samples_day*days_week
months_year = 12


def build_Attention(batch_size, n_timesteps, n_features, h1, h2, h3,f1,f2,k1,k2,l1,l2):
    input1 = keras.layers.Input(batch_shape=(batch_size,n_timesteps, n_features))
    #Spatial features
    conv1 = keras.layers.Conv1D(filters=int(f1), kernel_size=int(k1), activation='relu', strides=1, padding='same')(input1)
    conv2 = keras.layers.Conv1D(filters=int(f2), kernel_size=int(k2), activation='relu', strides=1, padding='same')(conv1)
    #Temporal features and Encoder
    encoder_out, encoder_hidden, encoder_cell = keras.layers.LSTM(int(l1), return_sequences=True, return_state=True)(conv2)
    #Weight encoded signal 
    attention = keras.layers.AdditiveAttention()([encoder_out, encoder_cell])        
    #Join attention weights with original encoded signal
    concatenate = tf.keras.layers.Concatenate()([encoder_out, attention])
    #Decode
    decoder = keras.layers.LSTM(int(l2))(concatenate)     
    #Gathered information interpretation through FNN 
    hidden1 = keras.layers.Dense(h1, activation='relu')(decoder)
    hidden2 = keras.layers.Dense(h2, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(h3, activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)     
    model = keras.models.Model(inputs=input1, outputs=output)
    model.compile(loss='mse', optimizer= 'adam')
    
    
    return model

def build_CNN(batch_size, n_timesteps, n_features, h1, h2, h3,f1, f2, k1, k2 ):
    ### input ###
    input1 = keras.layers.Input(batch_shape=(batch_size,n_timesteps, n_features))
    #Convolutional
    conv1 = keras.layers.Conv1D(filters=int(f1), kernel_size=int(k1), activation='relu', strides=1, padding='same')(input1)
    conv2 = keras.layers.Conv1D(filters=int(f2), kernel_size=int(k2), activation='relu', strides=1, padding='same')(conv1)
    flat = keras.layers.Flatten()(conv2)    
    hidden1 = keras.layers.Dense(h1, activation='relu')(flat)
    hidden2 = keras.layers.Dense(h2, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(h3, activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)
    model = keras.models.Model(inputs=input1, outputs=output)
     
    model.compile(loss='mse', optimizer= 'adam')    
    return model
    
def build_CNNLSTM(batch_size, n_timesteps, n_features, h1, h2, h3,f1,f2,k1,k2,l1):
    ### input ###
    input1 = keras.layers.Input(batch_shape=(batch_size,n_timesteps, n_features))
    #Convolutional
    conv1 = keras.layers.Conv1D(filters=int(f1), kernel_size=int(k1), activation='relu', strides=1, padding='same')(input1)
    conv2 = keras.layers.Conv1D(filters=int(f2), kernel_size=int(k2), activation='relu', strides=1, padding='same')(conv1)
    #Recurrent
    lstm1 = keras.layers.LSTM(int(l1), return_sequences=False, stateful=False)(conv2)
    hidden1 = keras.layers.Dense(h1, activation='relu')(lstm1)
    hidden2 = keras.layers.Dense(h2, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(h3, activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)
    model = keras.models.Model(inputs=input1, outputs=output)
    model.compile(loss='mse', optimizer= 'adam')    
     
    return model

def build_FNN(batch_size, n_timesteps, n_features, h1, h2, h3):
    
    ### input ###
    input1 = keras.layers.Input(shape=(n_timesteps,))
    hidden1 = keras.layers.Dense(int(h1), activation='relu')(input1)
    hidden2 = keras.layers.Dense(int(h2), activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(int(h3), activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)
    model = keras.models.Model(inputs=input1, outputs=output)
     
    model.compile(loss='mse', optimizer= 'adam')    
    return model

def build_LSTM(batch_size, n_timesteps, n_features, h1, h2, h3,l1):
    
    ### input ###
    input1 = keras.layers.Input(batch_shape=(batch_size,n_timesteps, n_features))
    #Recurrent
    lstm1 = keras.layers.LSTM(int(l1), return_sequences=False, stateful=False)(input1)
    
    
    hidden1 = keras.layers.Dense(h1, activation='relu')(lstm1)
    hidden2 = keras.layers.Dense(h2, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(h3, activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)
    model = keras.models.Model(inputs=input1, outputs=output)
    model.compile(loss='mse', optimizer= 'adam')  
     
    return model

##############################################
#############      START     #################
##############################################
    
#Architecture parameters
# config_device('gpu:3')
epoch = 150
batch_size = 672 #samples per week
n_timesteps = 5
n_features = 1 # as we use only one traffic reader for input

# Traffic reader IDs
espiras = ['145','418','137','169','157','259','217','106','295','318']
# Deep Architectures
architectures = ['FNN','CNN','LSTM','CNNLSTM', 'Attention']

for i in range(len(architectures)):
    df = pd.DataFrame()
    #Loop about 4 studied forecasting horizons t+1,t+2,t+3,t+4
    for h in range(4):    
        val_loss, train_loss, test_loss = list(), list(), list()
        with open('Trials/'+architectures[i]+'_trials_t+'+str(h+1)+'.pkl', 'rb') as f:
            pruebas = pickle.load(f)
        print('**********************Horizonte'+ str(h+1)+'***********************')
        num_params = list()
        
        
        
        for k in range(len(espiras)):
            
            
            #Best hyperparam config
            best = pruebas[k].best_trial    
            params = best['misc']['vals']
            
            if i== 0:
                model = build_FNN(batch_size, n_timesteps, n_features, np.array(params['h1']), 
                                  np.array(params['h2']), np.array(params['h3']))
            elif i== 1:
                model = build_CNN(batch_size, n_timesteps, n_features, 
                      np.array(params['h1']), np.array(params['h2']), np.array(params['h3']),
                       np.array(params['f1']), np.array(params['f2']),
                        np.array(params['k1']), np.array(params['k2']))
            elif i == 2:
                model = build_LSTM(batch_size, n_timesteps, n_features, 
                      np.array(params['h1']), np.array(params['h2']), np.array(params['h3']),
                       np.array(params['l1']))
            elif i == 3:
                model = build_CNNLSTM(batch_size, n_timesteps, n_features, 
                      np.array(params['h1']), np.array(params['h2']), np.array(params['h3']),
                       np.array(params['f1']), np.array(params['f2']),
                        np.array(params['k1']), np.array(params['k2']),
                        np.array(params['l1']))
            elif i == 4:
                model = build_Attention(batch_size, n_timesteps, n_features, 
                          np.array(params['h1']), np.array(params['h2']), np.array(params['h3']),
                           np.array(params['f1']), np.array(params['f2']),
                            np.array(params['k1']), np.array(params['k2']),
                            np.array(params['l1']), np.array(params['l2']))
                
                
                   
            
            #trainable params of each model
            trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
            # print('Trainable params of loop ' + espiras[k]+' : {:,}'.format(trainable_count))
            print(trainable_count)
            num_params.append(trainable_count)
        df['h = '+str(h+1)]=num_params
    
    
    for j in range(len(espiras)):
        df = df.rename(index={j : espiras[j]})   
        
    df.to_csv(architectures[i] + '_trainable_params.csv')   
    
    
print('Closing script...')

