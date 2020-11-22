import pyaudio
import numpy as np
import sys
import librosa  # sound analysis
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/

# CHUNK / RATE = Time of data to process
CHUNK = 2**15  # Bytes of data to process
RATE = 44100
TIME_WINDOW = CHUNK / RATE

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

for i in range(int(TIME_WINDOW * 100)):  # go for a few seconds
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    snd = parselmouth.Sound(data)  # Default sr = 44.1KHz
    intensity = snd.to_intensity()
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg()
    print("F1:", formant.get_value_at_time(1, TIME_WINDOW / 2))
    print("F2", formant.get_value_at_time(2, TIME_WINDOW / 2))
    print("intensity:", np.mean(intensity.values.T), "\npitch:", np.mean(pitch.selected_array['frequency']))
    sys.stdout.flush()

stream.stop_stream()
stream.close()
p.terminate()
