#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:09:48 2020

@author: eric
"""

import numpy as np
import pickle

algoritmos = ['FNN', 'CNN', 'LSTM', 'CNNLSTM', 'Attention']
# for h in range(4):    
train,validation, test = list(),list(), list()

for h in range(4):
    
    print(h)
    for i in range(len(algoritmos)):
        
        with open(algoritmos[i]+'_metrics_t+'+str(h+1)+'.pkl', 'rb') as f:
            pruebas = pickle.load(f)
        print('\n'+algoritmos[i])
        print('Train')
        print(np.array(pruebas['train']))
        print('Validation')
        print(np.array(pruebas['val']))
        print('Test')
        print(np.array(pruebas['test']))
        train.append(np.array(pruebas['train']))
        validation.append(np.array(pruebas['val']))
        test.append(np.array(pruebas['test']))
            
    
        
import csv
b = open('results_deep_NYC_train.csv', 'w')
a = csv.writer(b)
a.writerows(train)
b.close()
b = open('results_deep_NYC_validation.csv', 'w')
a = csv.writer(b)
a.writerows(validation)
b.close()
b = open('results_deep_NYC_test.csv', 'w')
a = csv.writer(b)
a.writerows(test)
b.close()
