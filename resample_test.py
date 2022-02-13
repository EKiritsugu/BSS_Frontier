# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:40:57 2022

resample test
@author: AR
"""

import numpy as np
import soundfile as sf
from scipy import signal
import librosa 
'''
fileway = 'E2A'
file1 = 'mixed/'+fileway+'_L.wav'
file2 = 'mixed/'+fileway+'_R.wav'
fs = 8000
tmp ,sr = librosa.load(file1 , sr = 16000)
tmp2 , sr = sf.read(file1)
minus = (tmp2[:,0]-tmp)/np.std(tmp)
y_8k = librosa.resample(tmp , sr , fs)

sf.write('after/test.wav',y_8k,fs)
'''
fileway = 'E2A'
file1 = 'mixed/'+fileway+'_L.wav'
file2 = 'mixed/'+fileway+'_R.wav'
tmp ,sr = sf.read(file1)
fs = 8000
#_,nch = np.shape(tmp)
'''
for i in range(2):
    tmp2 = librosa.resample(tmp[:,i] , sr ,fs)
    L_tmp2 = len(tmp2)
    tmp[:L_tmp2,i] = tmp2
Sig_out = tmp[:L_tmp2,:]
sf.write('after/resampletest.wav',Sig_out , 8000)
'''
def load_resampled_audio(fileway , fs):
    tmp ,sr = sf.read(file1)
    _,nch = np.shape(tmp)
    for i in range(nch):
        tmp2 = librosa.resample(tmp[:,i] , sr ,fs)
        L_tmp2 = len(tmp2)
        tmp[:L_tmp2,i] = tmp2
    Sig_out = tmp[:L_tmp2,:]
    return Sig_out

Sig_out = load_resampled_audio(file1,fs)
sf.write('after/resampletest2.wav',Sig_out , 8000)
