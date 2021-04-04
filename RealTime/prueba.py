import aubio
import numpy as num
import pyaudio
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from tkinter import TclError
import parselmouth as pm
from scipy.fftpack import fft
import struct

plt.ion()

# Some constants for setting the PyAudio and the
# Aubio.
CHUNK = 1024         # samples per frame
FORMAT = pyaudio.paInt16    # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
WAVE_OUTPUT_FILENAME = "file.wav"
METHOD                  = "default"
BUFFER_SIZE             = 2048
HOP_SIZE                = BUFFER_SIZE//2
PERIOD_SIZE_IN_FRAME    = HOP_SIZE

frames = []
fig = plt.figure(figsize=(5, 3))

# Initiating PyAudio object.
p = pyaudio.PyAudio()
# Open the microphone stream.
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Initiating Aubio's pitch detection object.
pDetection = aubio.pitch(METHOD, BUFFER_SIZE,
    HOP_SIZE, RATE)
# Set unit.
pDetection.set_unit("Hz")
# Frequency under -40 dB will considered
# as a silence.
pDetection.set_silence(-40)
max_int16 = 2**15

# Infinite loop!
while True:
    data = stream.read(PERIOD_SIZE_IN_FRAME)


    #Convertir datos para que aubio los entienda
    samples = num.fromstring(data,
       dtype=np.int16)
    #Hacer que sean floaat32 y normalizarlos
    float_sample = samples.astype(np.float32)
    sample_normalised = float_sample / max_int16 

    # Finally get the pitch.
    pitch = pDetection(sample_normalised)[0]

    #Aun no tengo claro como funciona el volumen
    volume = num.sum(samples**2)/len(samples)
    volume = "{:6f}".format(volume)
   

    # Finally print the pitch and the volume.
    print(str(pitch) + " " + str(volume))
    frames.append(data)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    except TclError:
   
        print('stream stopped')
        break