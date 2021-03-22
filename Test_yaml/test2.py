#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:38:35 2021

@author: louisbard
"""

## Test YAML ##

from sklearn.pipeline import make_pipeline

import yaml

import sys 

def from_yaml_to_func(method,params):
    prm = dict()
    if params!=None:
        for key,val in params.items():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


# load parameters file

yml = yaml.load(open('test2.yml'))



# imports 
for pkg, functions in yml['imports'].items():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters

seq_model2 = Sequential()




for item in yml['sequentiel']:
    
    for method,params in item.items():
        
        
       seq_model2.add(from_yaml_to_func(method,params))
       
       

seq_model2.compile(optimizer=yml['Compile']['optimizer'],loss=yml['Compile']['loss'])


       
        
        
import numpy as np
X_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))


epochs = yml['Fit']['epochs']

batch_size = yml['Fit']['batch_size']

seq_model2.fit(X_train,y_train, epochs=epochs, batch_size=batch_size)
        
            
            
            
        
    
        
        
        
        
        

