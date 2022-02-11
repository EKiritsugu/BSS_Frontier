import soundfile as sf
import numpy as np
from scipy.io import wavfile

# audio1,sr = sf.read('und/F1.wav')
# audio2,sr = sf.read('und/F2.wav')


gender = ['F','M']

for i in gender:
    audio = np.array([])
    for j in range(9):
        k = j+1
        filename = 'und/'+i+str(k)+'.wav'
        audio_tmp , sr = sf.read(filename)
        audio = np.hstack((audio,audio_tmp))
    sf.write('und/'+i+'.wav',audio,samplerate= sr)

