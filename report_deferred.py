import logging
import sys
import numpy as np
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
import os
import pandas as pd
from scipy import signal
from syllabe_nuclei import speech_rate
from jinja2 import Environment, FileSystemLoader
from flask import Flask, render_template, request, send_from_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json


def ibm_watson(file):
    with open("./sounds/" + file, "rb") as audio_file:
        authenticator = IAMAuthenticator("jf1ihfTWwiavbEwfUN1SYEgs_8UaOH5UyeZpzNy573sA")
        speech_to_text = SpeechToTextV1(
            authenticator=authenticator
        )

        speech_to_text.set_service_url("https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/7311c985-e03f-4d80-80d5-fcace75363b9")
        speech_to_text.set_disable_ssl_verification(True)
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            model='es-ES_BroadbandModel',
        )
        transcript = ""
        results = response.get_result()
        for result in results["results"]:
            transcript += result["alternatives"][0]["transcript"]
        return transcript


app = Flask(__name__)

@app.route('/base/<path:filename>')
def base_static(filename):
    return send_from_directory(app.root_path + 'sounds/', filename)

def draw_intensity(intensity, pitch):
    sns.set(rc={'figure.figsize':(12,4)}) # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 150 # Show images nicely
    plt.ylabel("Intensity [dB]")
    plt.xlabel("Seconds")
    plt.grid(True)
    plt.xlim([0, pitch.xmax])
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.savefig("static/img/intensity.png")
    plt.close()

    
    
def plot_wave(snd, pitch):
    sns.set(rc={'figure.figsize':(12,4)}) # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 150 # Show images nicely
    plt.ylabel("Amplitude")
    plt.xlabel("Seconds")
    plt.grid(True)
    plt.xlim([0, pitch.xmax])
    plt.plot(snd.xs(), snd.values.T)
    plt.savefig("static/img/wave.png")
    plt.close()
    
def plotOnGraph(pitch, color):
    sns.set(rc={'figure.figsize':(12,4)}) # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 150 # Show images nicely
    plt.ylim(0, 350)
    plt.ylabel("frequency [Hz]")
    plt.xlabel("Seconds")
    plt.grid(True)
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.xlim([0, pitch.xmax])
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2.5, color=color)
    plt.savefig("static/img/pitch.png")
    plt.close()
    


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
    file = request.args.get('file')
    file = file + ".wav"
    #file = "pulp-fiction.wav"
    transcript = ""#ibm_watson(file)

    snd = parselmouth.Sound('./sounds/' + file)
    csv = speech_rate(snd)
    csv = pd.DataFrame(csv.items())
    csv = csv.iloc[1:]

    intensity = snd.to_intensity()
    pitch = snd.to_pitch()

    b, a = signal.butter(3, 0.05)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, intensity.values[0][:], zi=zi*intensity.values[0][0])
    z1, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    intensity.values[0][:] = z1

    draw_intensity(intensity, pitch)
    plot_wave(snd, pitch)
    plotOnGraph(pitch, 'r')
    
    
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("report.html")

    template_vars = {"file": file, "transcripcion": transcript, "pausas" : csv.iloc[:3].to_html(), "mensaje": "Muchas pausas corrige x",
                    "intensidad": "intensity.png", "velocimetro": csv.iloc[3:].to_html(), "pitch": "pitch.png", "wave": "wave.png"}

    html_out = template.render(template_vars)
    return html_out


logging.basicConfig()



if __name__ == "__main__":
    app.run()