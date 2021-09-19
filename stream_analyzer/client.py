import concurrent.futures
import glob
import json
import logging
import os
import queue
import signal
import sys
import time

from scipy import signal as scipy_signal
import numpy as np
import parselmouth  # https://preaderselmouth.readthedocs.io/en/stable/
import pyaudio
import sounddevice as sa
from datetime import datetime
from numpy_ringbuffer import RingBuffer
from scipy.io import wavfile
from dotenv import load_dotenv
from syllabe_nuclei import speech_rate

"""
Installed on conda environment on W10
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf
"""
load_dotenv()
SECS = int(os.getenv('SECS',10))
CHUNK = 1024  # Bytes of data to process
RATE = 44100 // 2
BUFFER_SIZE = RATE * SECS  # BUFFER SIZE
FPS = 1  # Number of frames per seconds


np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.int16)
    global recorded_frames
    recorded_frames.put(data)
    buffer.extend(data)
    return in_data, pyaudio.paContinue


def remove_data():
    files = glob.glob("data/*")
    for f in files:
        os.remove(f)


def del_n_files(start, n):
    ini = time.time()
    files = np.arange(start, n)
    for f in files:
        n = str(f).zfill(4)
        try:
            os.remove(f"data/intensity_{n}.npy")
            os.remove(f"data/pitch_{n}.npy")
            os.remove(f"data/speech_rate_{n}.json")
        except:
            pass
    fin = time.time()
    print(f"Time to del files: {fin-ini}")


def signal_handler(signal=None, frame=None):
    print("\nprogram exiting gracefully")
    stream.stop_stream()
    stream.close()
    p.terminate()
    output = np.array(recorded_frames.queue, dtype=np.int16).flatten()
    timenow = datetime.now().strftime("%Y%m%d%H%M%S")
    wavfile.write(f"output/audio_{timenow}.wav", RATE, output)
    remove_data()
    sys.exit(0)


def get_sound(buff):
    return parselmouth.Sound(buff, sampling_frequency=RATE)


def get_intensity(snd):
    intensity = snd.to_intensity()
    return intensity


def get_pitch(snd):
    return snd.to_pitch()


def moving_average(x: np.ndarray, w: int):
    return np.convolve(x, np.ones(w), "valid") / w


def smooth_outliers(data):
    b, a = scipy_signal.butter(3, 0.05)
    y = scipy_signal.filtfilt(b, a, data)
    return y


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

p = pyaudio.PyAudio()
buffer = RingBuffer(capacity=(BUFFER_SIZE), dtype=np.float64)
buffer.extend(np.zeros(shape=BUFFER_SIZE))
recorded_frames = queue.Queue()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    output=False,
    frames_per_buffer=CHUNK,
    stream_callback=callback,
    # input_device_index=1,
)
time.sleep(1)

time_elapsed = 1
start_time = 0
last_update = 0
start_time = time.time()
n_files = 0
while time_elapsed <= 1800:  # go for 30mins
    if (time.time() - last_update) > (1.0 / FPS):
        last_update = time.time()
        buff = np.array(buffer)
        snd = get_sound(buff)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor.submit(del_n_files, n_files - 3, n_files - 2)
            future_srate = executor.submit(speech_rate, snd)
            future_intensity = executor.submit(get_intensity, snd)
            future_pitch = executor.submit(get_pitch, snd)
            # ------------------------------------------------------ #
            intensity = future_intensity.result()
            pitch = future_pitch.result()
            srate = future_srate.result()

        int_values: np.ndarray = intensity.values
        # int_values = smooth_outliers(int_values)
        pitch_values: np.ndarray = pitch.selected_array["frequency"]
        pitch_values = smooth_outliers(pitch_values)
        # pitch_values = moving_average(pitch_values, 20)
        n_files_str = str(n_files).zfill(4)
        ini = time.time()
        np.save(f"data/intensity_{n_files_str}", int_values)
        np.save(f"data/pitch_{n_files_str}", pitch_values)
        with open(f"data/speech_rate_{n_files_str}.json", "w") as outfile:
            json.dump(srate, outfile)

        time_elapsed = time.time() - start_time
        n_files += 1
        print(snd.duration)
        print("Time to run loop:", time.time() - last_update)
        print("Time elapsed:", time_elapsed)
signal_handler()
