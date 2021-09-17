import logging
import os
import sys

import numpy as np
import pandas as pd
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
import pyaudio
from syllabe_nuclei import speech_rate

"""
Installed on conda environment on W10
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

"""


logging.basicConfig(
    level=logging.INFO,
    filename="./logs/formants.log",
    filemode="w",
    format="%(message)s",
)

# CHUNK / RATE = Time of data to process
CHUNK = 2 ** 15  # Bytes of data to process (For speech_rate need 2**18 aprox)
RATE = 44100
TIME_WINDOW = CHUNK / RATE
SEC = 5 / TIME_WINDOW
ISWAVFILE = True
print(TIME_WINDOW)


def log_snd(snd, i):
    intensity = snd.to_intensity()
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg()
    f1 = formant.get_value_at_time(1, i)
    f2 = formant.get_value_at_time(2, i)
    intval = np.mean(intensity.values)
    logging.info("F1={}".format(f1))
    logging.info("F2={}".format(f2))
    logging.info("Intensity={}".format(np.mean(intensity.values - 20)))
    logging.info("Pitch={}".format(np.mean(pitch.selected_array["frequency"])))
    logging.info("time_windows={}".format(i))


if ISWAVFILE:
    for file in os.listdir("./sounds"):
        if file.endswith(".wav"):
            print(file)
            datasnd = parselmouth.Sound("./sounds/" + file)
            for i in np.arange(
                0.0, datasnd.duration, TIME_WINDOW
            ):  # go for a few seconds
                snd = datasnd.extract_part(from_time=i, to_time=i + TIME_WINDOW)
                csv = speech_rate(snd)
                logging.info(csv)

                intensity = snd.to_intensity()
                pitch = snd.to_pitch()
                formant = snd.to_formant_burg(time_step=0.05)
                for j in np.arange(0.0, TIME_WINDOW, 0.05):
                    log_snd(snd, j)


else:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    time_elapsed = 0
    while time_elapsed < 10:  # go for a few seconds
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        snd = parselmouth.Sound(data)  # Default sr = 44.1KHz)
        logging.info(speech_rate(snd))
        intensity = snd.to_intensity()
        pitch = snd.to_pitch()
        formant = snd.to_formant_burg(time_step=0.5)
        for j in np.arange(0, TIME_WINDOW, 0.5):
            log_snd(snd, j)
        time_elapsed += TIME_WINDOW
        sys.stdout.flush()
    p.terminate()
    stream.stop_stream()
    stream.close()

print("Ending...")
