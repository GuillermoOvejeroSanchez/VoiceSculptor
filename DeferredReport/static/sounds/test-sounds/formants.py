import math
import sys
import wave

import librosa
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.signal import hamming, lfilter

# Estimate formants using LPC.


def get_formants(file_path, ax):

    # Read from file.
    spf = wave.open(
        file_path, "r"
    )  # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav

    Fs = spf.getframerate()  # 44100 Hz
    ncoeff = 2 + Fs / 1000
    print(Fs)

    # Get file as numpy array.
    x = spf.readframes(-1)
    x = np.frombuffer(x, np.dtype(np.int16))

    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1.0, 0.63], x1)

    # Get LPC.
    A = librosa.lpc(x1, round(ncoeff))

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    frqs = sorted(angz * (Fs / (2 * math.pi)))
    vowels = ["A", "E", "I", "O", "U"]
    vowels_f1 = [650, 450, 300, 450, 330]
    vowels_f2 = [1300, 1800, 2200, 1000, 950]

    plt.scatter(
        vowels_f2,
        vowels_f1,
        marker="o",
        c="red",
    )
    for i, txt in enumerate(vowels):
        ax.annotate(txt, (vowels_f2[i] + 0.5, vowels_f1[i] + 0.5))
    plt.xlim(800, 2400)
    plt.ylim(150, 750)
    scatter = plt.scatter(frqs[2], frqs[1])
    ax = scatter.axes
    ax.invert_xaxis()
    ax.invert_yaxis()

    return frqs


files = ["O.wav", "I.wav", "A.wav"]
fig, ax = plt.subplots()
for f in files:
    print(get_formants(f, ax)[0:4])

plt.show()
