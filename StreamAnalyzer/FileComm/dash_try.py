import socket
from collections import deque

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from numpy_ringbuffer import RingBuffer
import glob
import random

# ---- Configuration ----
CHUNK = 1024  # Bytes of data to process
RATE = 44100 // 2
SECS = 15
BUFFER_SIZE = RATE * SECS  # BUFFER SIZE
FPS = 1  # Number of frames per seconds
i = 0
# -----------------------


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            className="container-fluid",
            children=[
                html.H1(
                    "Voice Analyzer", style={"color": "#000000", "text-align": "center"}
                )
            ],
        ),
        html.Div(dcc.Graph(id="gauge", animate=True)),
        html.Div(
            [
                html.H2(
                    "Intensity",
                    style={
                        "color": "#000000",
                    },
                ),
                dcc.Graph(id="int-graph", animate=True),
                html.H2("Pitch", style={"color": "#000000"}),
                dcc.Graph(id="pitch-graph", animate=True),
            ],
            className="row",
        ),
        dcc.Interval(id="graph-update", interval=FPS * 1000, n_intervals=0),
    ]
)

seconds = deque(maxlen=2)
seconds.append(1)

import json


def load_json(filename):
    with open(filename, "r") as json_file:
        srate = json.load(json_file)
    return srate


def get_field(srate: dict, field_name: str):
    return srate.get(field_name)


def get_data(f_type="intensity"):
    file = max(glob.glob(f"data/{f_type}_*"))
    interval = seconds[-1] + 1
    data: np.ndarray = np.load(file)
    seconds.append(interval)
    first = max(0, interval - SECS)
    interval = max(SECS, interval)
    Y = data.flatten()
    X = np.linspace(first, interval, num=Y.shape[0])
    return X, Y


@app.callback(Output("int-graph", "figure"), [Input("graph-update", "n_intervals")])
def update_graph_scatter(n):
    X, Y = get_data(f_type="intensity")
    data = plotly.graph_objs.Scatter(x=list(X), y=list(Y), name="Scatter", mode="lines")
    return {
        "data": [data],
        "layout": go.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[100, 170]),
        ),
    }


@app.callback(Output("pitch-graph", "figure"), [Input("graph-update", "n_intervals")])
def update_graph_scatter(n):
    X, Y = get_data(f_type="pitch")
    data = plotly.graph_objs.Scatter(x=list(X), y=list(Y), name="Scatter", mode="lines")
    y_max = max(Y)
    return {
        "data": [data],
        "layout": go.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[min(Y), max(170, y_max)]),
        ),
    }


@app.callback(Output("gauge", "figure"), [Input("graph-update", "n_intervals")])
def update_gauge(n):
    file = max(glob.glob(f"data/speech_rate_*"))
    srate = load_json(file)
    speed = get_field(srate, "speechrate(nsyll / dur)")
    print(speed)
    fig_gauge = go.Figure(
        go.Indicator(
            domain={"x": [0, 1], "y": [0, 1]},
            value=speed,
            mode="gauge+number+delta",
            title={"text": "Speed"},
            delta={"reference": 3.5},
            gauge={
                "axis": {"range": [0, 7]},
                "steps": [
                    {"range": [0, 1], "color": "#FFA500"},
                    {"range": [5, 7], "color": "#FF4500"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 5,
                },
            },
        )
    )
    return fig_gauge


if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port="8080")
