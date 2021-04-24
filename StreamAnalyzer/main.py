import logging
import sys
import numpy as np
from numpy.core.fromnumeric import shape
from numpy_ringbuffer import RingBuffer
from numpy.core.numeric import full
import parselmouth  # https://preaderselmouth.readthedocs.io/en/stable/
import pyaudio
import os
import time
import pandas as pd
from syllabe_nuclei import speech_rate
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

"""
Installed on conda environment on W10
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

"""


np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.INFO,filename="data.log" ,format="%(message)s")

CHUNK = 1024  # Bytes of data to process
RATE = 44100 // 2
SECS = 20
BUFFER_SIZE = RATE * SECS # BUFFER SIZE 

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.int16)
    logging.info(data)
    buffer.extend(data)
    return in_data, pyaudio.paContinue



plt.style.use('ggplot')
def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.5):
    if line1==None:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,alpha=0.8, linewidth=3)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1


def output_info(snd, line1):
    if snd.duration > 0.064:
        #print(speech_rate(sound=snd))
        intensity = snd.to_intensity()
        pitch = snd.to_pitch()
        print(np.mean(intensity))
        print(np.mean(pitch.selected_array['frequency']))
        line1 = live_plotter(intensity.xs(), intensity.values.T, line1)
        return line1




fps = 60 # Number of frames per seconds
time_elapsed = 0
start_time = 0
last_update = 0

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=callback
)
buffer = RingBuffer(capacity=(BUFFER_SIZE), dtype=np.int16)
start_time = time.time()
line1 = []
while time_elapsed <= 30:  # go for a few seconds
    if (time.time() - last_update) > (1./fps):
        last_update = time.time()
        ini = time.time()
        buff = np.array(buffer)
        snd = parselmouth.Sound(buff, sampling_frequency=RATE)
        print(snd.duration)
        line1 = output_info(snd, line1)
        ### Time Elapsed ###
        time_elapsed = time.time() - start_time
        print("Time elapsed:", time_elapsed)

