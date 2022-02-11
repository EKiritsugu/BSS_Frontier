# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:25:10 2022

@author: AR
"""
import numpy as np
import soundfile as sf
from scipy import signal
import time

nsources = 2#还是人为设置一个参数吧，表示信号源的数
fileway = 'E2A'


##############################################
file1 = 'mixed/'+fileway+'_L.wav'
file2 = 'mixed/'+fileway+'_R.wav'

tmp ,sr = sf.read(file1)
Sig_ori = np.zeros([nsources,len(tmp)])# 此处需要设定参数

tmp ,_= sf.read(file1)
Sig_ori[:,:] = tmp.T
tmp ,_= sf.read(file2)
Sig_ori[:,:] = tmp.T +Sig_ori[:,:]

##stft
_,_,Zxx0 = signal.stft(Sig_ori[0,:] , nperseg=2048 , noverlap=1536)
a,b = np.shape(Zxx0)

Sw = np.zeros((nsources,b,a) , dtype=complex)
for i in range(nsources):
    _,_,Zxx = signal.stft(Sig_ori[i,:] , nperseg=2048 , noverlap=1536)
    Sw[i,:,:] = Zxx.T 

#### analysis
X = Sw

[nmic,N,nfreq] = np.shape(X)

#X represents the signal after stft

eta = 0.1#learning rate
maxiter = 1000# number of iterations
tol = 1e-6 #When the difference of objective is less than tol, the algorithm terminates
nsou = nmic #number of sources

epsi=1e-6
pObj=float('inf')

W = np.zeros((nsou,nsou,nfreq),dtype = complex)
Wp = np.zeros((nsou,nsou,nfreq),dtype = complex)
dWp = np.zeros(np.shape(Wp),dtype = complex)
Q = np.zeros((nsou,nmic,nfreq),dtype = complex)
Xp = np.zeros((nsou,N,nfreq),dtype = complex)

S = np.zeros((nsou,N,nfreq),dtype = complex)
Ssq = np.zeros((nsou,N),dtype = complex)
Ssq1 = np.zeros((nsou,N),dtype = complex)

# PCA
# PCA
for p in range(nfreq):
    Xmean = np.mean(X[:,:,p],1)
    Xmean = Xmean.reshape(len(Xmean),1)
    Xmean = Xmean @ np.ones((1,N))

    Rxx = (X[:,:,p]- Xmean) @ (X[:,:,p]- Xmean).T.conjugate() /N

    D,E = np.linalg.eig(Rxx)
    

    d = np.real(D)
    tmp = np.sort(-d)
    order = np.argsort(-d)

    E = E[:,order[:nsou]]

    d = d[order[:nsou]]
    d = np.real(d)
    d = np.power(d , -0.5)
    D2 = np.diag(d)
    Q[:,:,p] = D2@E.T.conjugate()#

    Xp[:,:,p] = Q[:,:,p]@(X[:,:,p]-Xmean)
    Wp[:,:,p] = np.eye(nsou)

# independent vector ayalysis

for iter in range(maxiter):

    dlw = 0
    for k in range(nfreq):
        S[:,:,k] = Wp[:,:,k] @ Xp[:,:,k]

    Ssq = np.sum(np.power(np.abs(S),2),2)
    Ssq = np.sqrt(Ssq)
    Ssq1 = np.power(Ssq + epsi , -1)

    for k in range(nfreq):
        Phi = Ssq1 * S[:,:,k]
        dWp[:,:,k] = (np.eye(nsou) - Phi @ S[:,:,k].T.conjugate()/N ) @ Wp[:,:,k]
        dlw = dlw + np.log(epsi + np.abs(np.linalg.det(Wp[:,:,k])) )

    # update
    Wp = Wp + eta * dWp

for k in range(nfreq):
    W[:,:,k] = Wp[:,:,k] @ Q[:,:,k]
    W[:,:,k] = np.diag(np.diag(np.linalg.pinv(W[:,:,k]))) @ W[:,:,k]

for k in range(nfreq):
    S[:,:,k] = W[:,:,k]@X[:,:,k]


Sw_hat = S


## istft


_ , tmp = signal.istft(Sw_hat[0,:,:].T, nperseg=2048 , noverlap=1536)
St_hat = np.zeros((nsources , len(tmp)))
for i in range(nsources):
    _ , tmp = signal.istft(Sw_hat[i,:,:].T, nperseg=2048 , noverlap=1536)
    St_hat[i,:] = np.real(tmp)


sf.write('after/'+fileway+'.wav', St_hat.T,samplerate= sr)




