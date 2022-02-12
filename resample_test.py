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

fileway = 'E2A'
file1 = 'mixed/'+fileway+'_L.wav'
file2 = 'mixed/'+fileway+'_R.wav'

tmp ,sr = librosa.load(file1 , sr = 16000)
y_8k = librosa.resample(tmp , sr , 8000)

sf.write('test.wav',y_8k,8000)


