import logging
import sys
import numpy as np
import parselmouth  # https://parselmouth.readthedocs.io/en/stable/
import os
import pandas as pd
from syllabe_nuclei import speech_rate
from jinja2 import Environment, FileSystemLoader
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/report")
def report():
    username = request.args.get('name')
    return render_template(html_out, template_vars)


logging.basicConfig()
file = "pulp-fiction.wav"
snd = parselmouth.Sound('./sounds/' + file)
csv = speech_rate(snd)
print(csv)
csv = pd.DataFrame(csv.items())
csv.to_csv('report.csv')
intensity = snd.to_intensity()
pitch = snd.to_pitch()
logging.info("Intensity={}".format(np.mean(intensity.values - 20)))
logging.info("Pitch={}".format(np.mean(pitch.selected_array['frequency'])))

env = Environment(loader=FileSystemLoader('.'))
template = env.get_template("report.html")

template_vars = {"transcripcion": "transcripcion entera del audio", "pausas" : "20", "mensaje": "Muchas pausas corrige x",
                 "intensidad": intensity, "velocimetro": "120", "pitch": pitch}

html_out = template.render(template_vars)

if __name__ == "__main__":
    app.run()