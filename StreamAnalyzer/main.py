import logging
import sys
import signal
import numpy as np
from numpy.core.fromnumeric import shape
from numpy_ringbuffer import RingBuffer
from numpy.core.numeric import full
import parselmouth  # https://preaderselmouth.readthedocs.io/en/stable/
import pyaudio
import sounddevice as sa
import os
import time
import pandas as pd
from syllabe_nuclei import speech_rate
from threading import Thread
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
import pickle
import redis
from scipy.io import wavfile
import queue

"""
Installed on conda environment on W10
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

"""


np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.INFO, format="%(message)s")

CHUNK = 1024  # Bytes of data to process
RATE = 44100 // 2
SECS = 5
BUFFER_SIZE = RATE * SECS  # BUFFER SIZE
REDIS_PITCH = "PITCH"
REDIS_INTENSITY = "INTENSITY"


def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.int16)
    global recorded_frames
    recorded_frames.put(data)
    buffer.extend(data)
    return in_data, pyaudio.paContinue


def signal_handler(signal=None, frame=None):
    print("\nprogram exiting gracefully")
    stream.stop_stream()
    stream.close()
    p.terminate()
    output = np.array(recorded_frames.queue, dtype=np.int16).flatten()
    wavfile.write("example.wav", RATE, output)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

fps = 30  # Number of frames per seconds
time_elapsed = 0
start_time = 0
last_update = 0

r = redis.Redis()
r.flushall()
p = pyaudio.PyAudio()
buffer = RingBuffer(capacity=(BUFFER_SIZE), dtype=np.int16)
recorded_frames = queue.Queue()

for device_index in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(device_index)
    if device_info["maxInputChannels"] > 0:
        print(device_info)

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    output=False,
    frames_per_buffer=CHUNK,
    stream_callback=callback,
    input_device_index=1,
)
time.sleep(1)

start_time = time.time()
while time_elapsed <= 40:  # go for a few seconds
    if (time.time() - last_update) > (1.0 / fps):
        last_update = time.time()
        ini = time.time()
        buff = np.array(buffer)
        snd = parselmouth.Sound(buff, sampling_frequency=RATE)
        print(snd.duration)
        intensity = snd.to_intensity(50)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        ### Time Elapsed ###
        r.xadd(REDIS_INTENSITY, {"intensity": intensity.values.T.tobytes(), "seconds": intensity.xs().tobytes()})
        r.xadd(REDIS_PITCH, {"pitch": pitch_values.tobytes(), "seconds": pitch.xs().tobytes()})
        # time.sleep(1)
        time_elapsed = time.time() - start_time
        # print("Time elapsed:", time_elapsed)
signal_handler()
