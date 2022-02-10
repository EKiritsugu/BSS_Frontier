# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:10:46 2022

@author: AR
"""
import pyroomacoustics as pra


import numpy as np
import soundfile as sf


rt60 = 0.300
room_dim = [7 , 5, 2.75]

e_absorption , max_order = pra.inverse_sabine(rt60 , room_dim)


room = pra.ShoeBox(room_dim , materials = pra.Material(e_absorption) , max_order = max_order)


mic_locs = np.c_[
    [3.03,1,1.5],
    [3.0 - 0.03,1,1.5]
]



room.add_microphone_array(mic_locs)



### 4组输入时候的代码
audio1,sr = sf.read('und/F.wav')
#audio2,sr = sf.read('ori/Synth_2.wav')

###4组输入时候的麦克风位置

#1号
room.add_source([1.71442,2.53209, 1.5] , signal = audio1, delay = 0)
#2号
#room.add_source([4.28558,2.53209, 1.5] , signal = audio2, delay = 0)

room.simulate()

simulation_data = room.mic_array.signals

#_,nchannels,sampwidth , framerate, nframes = getdata('reportmono01.wav')

# for i in range(2):
#     k = i+1
#     savewav('mixed/6.8b3_'+str(k)+'.wav', simulation_data[i,:] )
sf.write('mixed/E2A_L.wav',simulation_data.T,samplerate= sr)





