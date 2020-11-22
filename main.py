import pyaudio
import numpy as np
import sys
# import librosa
import parselmouth

CHUNK = 2**14
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

for i in range(int(10*44100/1024)):  # go for a few seconds
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    snd = parselmouth.Sound(data)  # Default sr = 44.1KHz
    intensity = snd.to_intensity()
    pitch = snd.to_pitch()
    print("intensity:", intensity, "\npitch:", pitch)
    sys.stdout.flush()

stream.stop_stream()
stream.close()
p.terminate()
