#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:32:29 2021

@author: louisbard
"""

from numpy import *


import numpy as np
import glob
import re
#from pylab import *
from scipy.signal import *
import pandas as pd


def bandpass(sig,band,fs):
    
    B,A = butter(5, array(band)/(fs/2), btype='bandpass')
    
    return lfilter(B, A, sig, axis=0)




for test in [False,True] :
    
     prefix = '' if test is False else 'test_'
     
     DataFolder = '../data/train/' if test is False else '../data/test/'
     
     list_of_files = glob.glob(DataFolder + 'Data_*.csv')
     
     list_of_files.sort()
     
     reg = re.compile('\d+')
     
     freq = 200.0

     epoc_window = 1.3*freq
     
     X = []
     
     for f in list_of_files:
         
        print(f)
        
        user,session = reg.findall(f)
        
        sig = np.array(pd.io.parsers.read_csv(f))

        EEG = sig[:,1:-2]
        
        EOG = sig[:,-2]
        
        Trigger = sig[:,-1]

        sigF = bandpass(EEG,[1.0,40.0],freq)
    
    
        idxFeedBack = np.where(Trigger==1)[0]
        
        for fbkNum,idx in enumerate(idxFeedBack):
            
            
            
            X.append(sigF[idx:int(idx+epoc_window),:])
            
     X = array(X).transpose((0,2,1))
            
     save(prefix + 'epochs.npy',X)
            
     
     
     
     
