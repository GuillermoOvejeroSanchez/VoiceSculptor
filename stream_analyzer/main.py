import glob
from collections import deque
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import subprocess
import plotly
import plotly.graph_objs as go
import plotly.io as pio
from dash.dependencies import Input, Output
from dotenv import load_dotenv
from numpy_ringbuffer import RingBuffer
import json
pio.templates.default = "seaborn"

load_dotenv()
# ---- Configuration ----
SECS = int(os.getenv('SECS',10))
CHUNK = 1024  # Bytes of data to process
RATE = 44100 // 2
BUFFER_SIZE = RATE * SECS  # BUFFER SIZE
FPS = 1  # Number of frames per seconds

# Style
TEXT_STYLE = {"textAlign": "center", "color": "#191970"}
CONTENT_STYLE = {
    "margin-left": "5%",
    "margin-right": "5%",
    "top": 0,
    "padding": "20px 10px",
}
####################

client_process = None
app = dash.Dash(__name__)
srate_row = dbc.Row(
    [
        dbc.Col(dcc.Graph(id="pauses", animate=True)),
        dbc.Col(dcc.Graph(id="gauge", animate=True)),
        dbc.Col(dcc.Graph(id="nsyll", animate=True)),
    ]
)
intensity_row = dbc.Row(
    [
        html.H2(
            "Intensity",
            style=TEXT_STYLE,
        ),
        dcc.Graph(id="intensity-graph", animate=True),
    ]
)
pitch_row = dbc.Row(
    [
        html.H2(
            "Pitch",
            style=TEXT_STYLE,
        ),
        dcc.Graph(id="pitch-graph", animate=True),
    ]
)
content = html.Div(
    [
        html.H2("Voice Analyzer", style=TEXT_STYLE),
        html.Div(
            html.Button("Play", id="play", n_clicks=0),
            style={"display": "inline-block"},
        ),
        html.Hr(),
        srate_row,
        intensity_row,
        pitch_row,
        dcc.Interval(id="graph-update", interval=FPS * 1000, n_intervals=0),
    ],
    style=CONTENT_STYLE,
)
app.layout = content


def start_deque():
    seconds = deque(maxlen=2)
    seconds.append(1)
    return seconds


seconds = start_deque()


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


def remove_data():
    files = glob.glob("data/*")
    for f in files:
        os.remove(f)


def return_intensity(n):
    X, Y = get_data(f_type="intensity")
    data = plotly.graph_objs.Scatter(x=list(X), y=list(Y), name="Scatter", mode="lines")
    return {
        "data": [data],
        "layout": go.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[100, 180]),
        ),
    }


def return_pitch(n):
    X, Y = get_data(f_type="pitch")
    data = plotly.graph_objs.Scatter(x=list(X), y=list(Y), name="Scatter", mode="lines")
    return {
        "data": [data],
        "layout": go.Layout(
            xaxis=dict(range=[min(X), max(X)]), yaxis=dict(range=[0, max(200, max(Y))])
        ),
    }


@app.callback(Output("play", "children"), [Input("play", "n_clicks")])
def play_pause(n):
    global client_process
    if n == 0:
        remove_data()
        return "Play"
    elif n % 2 == 0 and client_process is not None:
        client_process.terminate()
        remove_data()
        return "Play"
    else:
        print("now is playing, hit again to stop")
        print(os.name)
        remove_data()
        global seconds
        seconds = start_deque()
        client_process = subprocess.Popen(args=["python","client.py"])
        return "Stop"


@app.callback(
    Output("intensity-graph", "figure"),
    Output("pitch-graph", "figure"),
    [Input("graph-update", "n_intervals")],
)
def update_graph_scatter(n):
    return return_intensity(n), return_pitch(n)


@app.callback(
    Output("gauge", "figure"),
    Output("pauses", "figure"),
    Output("nsyll", "figure"),
    [Input("graph-update", "n_intervals")],
)
def update_first_row(n):
    file = max(glob.glob(f"data/speech_rate_*"))
    print(file)
    srate = load_json(file)
    speed = get_field(srate, "speechrate(nsyll / dur)")
    pause = get_field(srate, "npause")
    syll = get_field(srate, "nsyll")
    print(speed, pause, syll)

    fig_gauge = go.Figure(
        go.Indicator(
            domain={"x": [0, 1], "y": [0, 1]},
            value=speed,
            mode="gauge+number+delta",
            title={"text": "Speed"},
            delta={"reference": 4.25},
            gauge={
                "axis": {"range": [0, 8]},
                "steps": [
                    {"range": [3.5, 4], "color": "#87ca66"},
                    {"range": [4, 4.5], "color": "#66CA6C"},
                    {"range": [4.5, 5], "color": "#87ca66"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.8,
                    "value": 5.5,
                },
            },
        )
    )
    fig_pause = go.Figure(
        go.Indicator(mode="number", value=pause, title={"text": "Pauses"})
    )
    fig_syll = go.Figure(
        go.Indicator(mode="number", value=syll, title={"text": "Nº Syllabes"})
    )
    return fig_gauge, fig_pause, fig_syll


if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="127.0.0.1",
        port="5050",
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )
