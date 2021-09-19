import logging
import matplotlib
from scipy import signal as scipy_signal
import pandas as pd
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
from flask import Flask, request, redirect, url_for
from jinja2 import Environment, FileSystemLoader
from syllabe_nuclei import speech_rate
from pathlib import Path

matplotlib.use("Agg")
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
import plotly


def init_app():
    app = Flask(__name__)
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
        print(results)
        for result in results["results"]:
            transcript += result["alternatives"][0]["transcript"]
            transcript = transcript[:-1] + ". "
        return transcript


def smooth_outliers(data):
    b, a = scipy_signal.butter(3, 0.05)
    y = scipy_signal.filtfilt(b, a, data)
    return y


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


@app.route("/")
def index():
    env = Environment(loader=FileSystemLoader("."))
    template_path = "index.html"
    template = env.get_template(str(template_path))
    html_out = template.render()
    return html_out


@app.route("/", methods=["POST"])
def upload_file():
    uploaded_file = request.files["file"]
    if uploaded_file.filename != "":
        uploaded_file.save("static/sounds/" + uploaded_file.filename)
    return redirect(url_for(f"report", file=uploaded_file.filename))


@app.route("/report")
def report():
    file = request.args.get("file")
    if file[-4:] not in {".wav",".mp3"}:
        file = file + ".wav"
    transcript = ibm_watson(file)

    path_file = path_sounds / file
    print(path_file.absolute())
    snd = parselmouth.Sound(str(path_file))
    csv = speech_rate(snd)
    print(csv)
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
