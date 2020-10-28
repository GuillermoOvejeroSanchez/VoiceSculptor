import pyaudio
import numpy as np
import sys
import librosa

CHUNK = 2**11
RATE = 44100

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)

for i in range(int(10*44100/1024)): #go for a few seconds
    data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
    peak=np.average(np.abs(data))*2
    lpcs = librosa.lpc(data)
    bars="#"*int(500*peak/2**16)
    print(lpcs)
    print("{:04d} {:04.00f} {}".format(i,peak,bars))
    sys.stdout.flush()

stream.stop_stream()
stream.close()
p.terminate()