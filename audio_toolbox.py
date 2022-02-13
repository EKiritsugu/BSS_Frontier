# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:18:04 2022

Audio toolbox
@author: AR
"""
import numpy as np
import soundfile as sf
from scipy import signal
import librosa 

def load_resampled_audio(fileway , fs):
    '''
    This function accepts path-like object and samplying frequency. The returned value is an array of dowmsampled waveform.
    '''
    tmp ,sr = sf.read(fileway)
    _,nch = np.shape(tmp)
    for i in range(nch):
        tmp2 = librosa.resample(tmp[:,i] , sr ,fs)
        L_tmp2 = len(tmp2)
        tmp[:L_tmp2,i] = tmp2
    Sig_out = tmp[:L_tmp2,:]
    return Sig_out