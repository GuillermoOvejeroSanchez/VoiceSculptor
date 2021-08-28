import logging
import os
import pickle
import sys
import time
from threading import Thread

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth  # https://preaderselmouth.readthedocs.io/en/stable/
import pyaudio
import redis
import seaborn as sns
import sounddevice as sa
from matplotlib.animation import FuncAnimation
from numpy.core.fromnumeric import shape
from numpy.core.numeric import full
from numpy_ringbuffer import RingBuffer
from scipy import signal
from syllabe_nuclei import speech_rate

"""
Installed on conda environment on W10
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

"""

REDIS_PITCH = "PITCH"
REDIS_INTENSITY = "INTENSITY"
r = redis.Redis()

while True:
    streaming = r.xread({REDIS_PITCH: "$"}, None, 0)
    # r.xdel(REDIS_INTENSITY, streaming[0][1][0][0])
    r.xdel(REDIS_PITCH, streaming[0][1][0][0])
    # intensity = np.frombuffer(streaming[0][1][0][1][b"intensity"])
    # seconds = np.frombuffer(streaming[0][1][0][1][b'seconds'])
    intensity = np.frombuffer(streaming[0][1][0][1][b"pitch"])
    seconds = np.frombuffer(streaming[0][1][0][1][b"seconds"])
    plt.cla()
    x = np.arange(0, 10)
    y = np.full((10, 1), 160)
    plt.ylim(90, 270)
    # plt.xlim(10,20)
    plt.plot(x, y)
    plt.scatter(seconds, intensity, marker="o")
    plt.draw()
    plt.pause(0.05)
