# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:39:53 2022

@author: AR

down sampled
with adaptive learning rate
generalized gaussian source prior
partly succeed
"""

import numpy as np
import soundfile as sf
from scipy import signal
import audio_toolbox as at
fs = 8000
eta0 = 0.1#learning rate
beta = 0.5
lamb = 0.999
nsources = 2#还是人为设置一个参数吧，表示信号源的数
fileway = 'E2A'
file1 = 'mixed/'+fileway+'_L.wav'
file2 = 'mixed/'+fileway+'_R.wav'

Sig_ori = at.load_resampled_audio(file1,fs)
Sig_ori = Sig_ori + at.load_resampled_audio(file2,fs)
Sig_ori = Sig_ori.T
##stft
_,_,Zxx0 = signal.stft(Sig_ori[0,:] , nperseg=2048 , noverlap=1536)
a,b = np.shape(Zxx0)
Sw = np.zeros((nsources,b,a) , dtype=complex)
for i in range(nsources):
    _,_,Zxx = signal.stft(Sig_ori[i,:] , nperseg=2048 , noverlap=1536)
    Sw[i,:,:] = Zxx.T 
del a,b,Zxx,Zxx0

#### analysis
[nmic,nframes,nfreq] = np.shape(Sw)    
X = Sw.swapaxes(0, 1)
xi = np.zeros(nfreq)
gk = np.zeros(nfreq)
gk_1 = np.zeros(nfreq)
eta = np.zeros(nfreq)
eta_1 = np.zeros(nfreq)
gk0 = np.ones(nfreq)*1e-5

tol = 1e-6 #When the difference of objective is less than tol, the algorithm terminates
nsou = nmic #number of sources
epsi=1e-6
S_out = np.zeros(np.shape(Sw),dtype = complex)

W = np.expand_dims(np.eye(nsou,dtype=complex) , 0).repeat(nfreq,axis = 0)
dW = np.zeros((nfreq,nsou,nsou),dtype = complex)

# online independent vector ayalysis
xn = np.zeros((nmic,nfreq),dtype=complex)
yn = np.zeros((nmic,nfreq),dtype=complex)
y_out = np.zeros((nmic,nfreq),dtype=complex)
for frame in range(nframes):
   
    xn = X[frame,:,:]#signal of n_frame
    
    for k in range(nfreq):
        yn[:,k] = W[k,:,:] @ xn[:,k]
        y_out = np.diag(np.diag(np.linalg.inv(W[k,:,:])))@yn[:,k]
        S_out[:,frame,k ] = y_out
#xi  
    xi = beta*xi + (1-beta) * np.sum(np.abs(xn)**2,axis = 0)/nmic
#Phi
    S = (np.sum(np.abs(yn)**2 , axis = 1))**-(2/3)
    for k in range(nfreq):
        Phi = np.expand_dims(yn[:,k] *S,axis = 1)
        Rk = Phi @ np.expand_dims(yn[:,k] .conjugate() ,axis = 0)
        Lambda = np.diag(np.diag(Rk))
        dW[k , : , :] = (Lambda - Rk ) @ W[k,:,:]
        # gk[k] eta[k]
        gk[k] = np.linalg.norm(Lambda - Rk)
        '''
        if frame == 0:
            #gk0[k] = gk[k]
            eta[k] = eta0
            gk_1[k] = gk[k]
            eta_1[k] = eta[k]
        else:
            '''
        eta[k] = eta0*(gk[k]/gk0[k])
        eta[k] = ((1-lamb)*eta[k] + lamb *eta_1[k])
        '''
        if eta[k]>100:
            eta[k] = 100
        if eta[k]<1:
            eta[k] = 1
            '''
        eta_1[k] = eta[k]
        gk_1[k] = gk[k] 
        W[k,:,:] = W[k,:,:] + eta[k] * (xi[k]**(-0.5)) * dW[k,:,:] 
    print(np.sum(eta))
        
## istft
_ , tmp = signal.istft(S_out[0,:,:].T, nperseg=2048 , noverlap=1536)
St_hat = np.zeros((nsources , len(tmp)))
for i in range(nsources):
    _ , tmp = signal.istft(S_out[i,:,:].T, nperseg=2048 , noverlap=1536)
    St_hat[i,:] = np.real(tmp)

sf.write('after/'+fileway+'_4.wav', St_hat.T,samplerate= fs)




