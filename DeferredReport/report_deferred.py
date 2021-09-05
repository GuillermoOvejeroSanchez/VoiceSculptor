import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
from flask import Flask, render_template, request, send_from_directory
from jinja2 import Environment, FileSystemLoader
from scipy import signal
from syllabe_nuclei import speech_rate
from pathlib import Path

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import seaborn as sns
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1

import dash
import dash_html_components as html
import plotly.graph_objects as go
import plotly


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(server=server, routes_pathname_prefix="/")
    dash_app.layout = html.Div(id="dash-container")

    return dash_app.server


def init_app():
    app = Flask(__name__)
    with app.app_context():
        app = init_dashboard(app)
    return app


app = init_app()

path_sounds = Path("static/sounds/")

# https://cloud.ibm.com/apidocs/speech-to-text?code=python
def ibm_watson(file):
    path_file = path_sounds / file
    with open(str(path_file), "rb") as audio_file:
        authenticator = IAMAuthenticator("jf1ihfTWwiavbEwfUN1SYEgs_8UaOH5UyeZpzNy573sA")
        speech_to_text = SpeechToTextV1(authenticator=authenticator)

        speech_to_text.set_service_url(
            "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/7311c985-e03f-4d80-80d5-fcace75363b9"
        )
        speech_to_text.set_disable_ssl_verification(True)
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type="audio/wav",
            model="es-ES_BroadbandModel",
            timestamps=True,
        )
        transcript = ""
        results = response.get_result()
        for result in results["results"]:
            transcript += result["alternatives"][0]["transcript"]
        return transcript


from scipy import signal as scipy_signal


def smooth_outliers(data):
    b, a = scipy_signal.butter(3, 0.05)
    y = scipy_signal.filtfilt(b, a, data)
    return y


# @app.route("/base/<path:filename>")
# def base_static(filename):
#     file_path = app.root_path + path_sounds
#     return send_from_directory(file_path, filename)


def plot_intensity(intensity, pitch):
    Y = intensity.values
    Y = Y.flatten()
    Y = smooth_outliers(Y)
    X = intensity.xs()
    data = [{"x": X, "y": Y}]
    div = plotly.offline.plot(data, include_plotlyjs=bool, output_type="div")
    return div


def plot_pitch(snd, pitch):
    Y = pitch.selected_array["frequency"]
    Y = smooth_outliers(Y)
    X = pitch.xs()
    data = [{"x": X, "y": Y}]
    div = plotly.offline.plot(data, include_plotlyjs=bool, output_type="div")
    return div


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


@app.route("/report")
def report():
    file = request.args.get("file")
    file = file + ".wav"
    transcript = ""  # ibm_watson(file)

    path_file = path_sounds / file
    print(path_file.absolute())
    snd = parselmouth.Sound(str(path_file))
    csv = speech_rate(snd)
    csv = pd.DataFrame(csv.items())
    csv = csv.iloc[1:]

    intensity = snd.to_intensity()
    pitch = snd.to_pitch()

    div_intensity = plot_intensity(intensity, pitch)
    div_pitch = plot_pitch(snd, pitch)

    env = Environment(loader=FileSystemLoader("."))
    template_path = "report.html"
    template = env.get_template(str(template_path))

    template_vars = {
        "file": file,
        "transcripcion": transcript,
        "pausas": csv.iloc[:3].to_html(),
        "intensity": div_intensity,
        "velocimetro": csv.iloc[3:].to_html(),
        "pitch": div_pitch,
    }

    html_out = template.render(template_vars)
    return html_out


logging.basicConfig()
if __name__ == "__main__":
    app.run()
