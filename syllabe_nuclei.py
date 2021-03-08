import logging
import sys
import numpy as np
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
import os
import pandas as pd
#from syllabe_nuclei import speech_rate
from jinja2 import Environment, FileSystemLoader
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)


def draw_intensity(intensity):
    sns.set()
    plt.rcParams['figure.dpi'] = 150
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(True)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def plotOnGraph(pitch, color):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2.5, color=color)
    
def setupGraph(ymin, ymax):
    sns.set() # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 150 # Show images nicely
    plt.figure()
    plt.ylim(ymin, ymax)
    plt.ylabel("frequency [Hz]")
    plt.xlabel("seconds")
    plt.grid(True)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/report")
def report():
    #file = request.args.get('file')
    #file = file + ".wav"
    #file = "pulp-fiction.wav"
    snd = parselmouth.Sound('file.wav')
    csv = speech_rate(snd)
    csv = pd.DataFrame(csv.items())
    csv = csv.iloc[1:]

    intensity = snd.to_intensity()
    plt.figure()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.savefig("static/intensity.png")

    pitch = snd.to_pitch()
    setupGraph(0, pitch.ceiling)
    plotOnGraph(pitch, 'r')
    plt.savefig("static/pitch.png")

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("report.html")

    template_vars = {"file": file, "transcripcion": "transcripcion entera del audio", "pausas" : csv.iloc[:3].to_html(), "mensaje": "Muchas pausas corrige x",
                    "intensidad": "intensity.png", "velocimetro": csv.iloc[3:].to_html(), "pitch": "pitch.png"}

    html_out = template.render(template_vars)
    return html_out


logging.basicConfig()



if __name__ == "__main__":
    app.run()