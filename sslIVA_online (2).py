# -*- coding: utf-8 -*-
"""
用于SSLIVA的在线算法，仅仅考虑输入源数量等于源型号数量的情况
succeed 
别动
"""
import numpy as np
import soundfile as sf
from scipy import signal

eta = 2#learning rate
beta = 0.5

nsources = 2#还是人为设置一个参数吧，表示信号源的数
fileway = 'E2A'
file1 = 'mixed/'+fileway+'_L.wav'
file2 = 'mixed/'+fileway+'_R.wav'

tmp ,sr = sf.read(file1)
Sig_ori = np.zeros([nsources,len(tmp)])# 此处需要设定参数
tmp ,_= sf.read(file1)
Sig_ori[:,:] = tmp.T
tmp ,_= sf.read(file2)
Sig_ori[:,:] = tmp.T +Sig_ori[:,:]
del tmp
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
    S = np.sqrt(np.sum(np.abs(yn)**2 , axis = 1))**-1
    for k in range(nfreq):
        Phi = np.expand_dims(yn[:,k] *S,axis = 1)
        Rk = Phi @ np.expand_dims(yn[:,k] .conjugate() ,axis = 0)
        Lambda = np.diag(np.diag(Rk))
        dW[k , : , :] = (Lambda - Rk ) @ W[k,:,:]
        W[k,:,:] = W[k,:,:] + eta*xi[k]**(-0.5) * dW[k , : , :] 
        
        
## istft
_ , tmp = signal.istft(S_out[0,:,:].T, nperseg=2048 , noverlap=1536)
St_hat = np.zeros((nsources , len(tmp)))
for i in range(nsources):
    _ , tmp = signal.istft(S_out[i,:,:].T, nperseg=2048 , noverlap=1536)
    St_hat[i,:] = np.real(tmp)

sf.write('after/'+fileway+'_ref.wav', St_hat.T,samplerate= sr)

