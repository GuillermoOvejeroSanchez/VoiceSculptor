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

# ---- Configuration ----
CHUNK = 1024  # Bytes of data to process
RATE = 44100 // 2
SECS = 10
BUFFER_SIZE = RATE * SECS  # BUFFER SIZE
FPS = 1  # Number of frames per seconds
HOST = "127.0.0.1"  # Symbolic name meaning all available interfaces
PORT = 2345  # Arbitrary non-privileged port
# list of readable sockets. s is readable if a client is waiting.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(2)
conn, addr = s.accept()
i = 0
# -----------------------


def process_data(data: bytes, i: int):
    data_np = np.frombuffer(data)
    return data_np
    # data_string: str = data.decode("utf-8")
    # data_dict: dict = loads(data_string)
    # if i % 10 == 0:
    #     nsyll += data_dict["nsyll"]
    # print(data_dict)
    # print(nsyll)


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            className="container-fluid",
            children=[html.H1("Voice Analyzer", style={"color": "#CECECE"})],
        ),
        html.Div(
            [dcc.Graph(id="live-graph", animate=True)],
            className="row",
        ),
        dcc.Interval(id="graph-update", interval=FPS * 1000, n_intervals=0),
    ]
)

seconds = deque(maxlen=2)
seconds.append(1)


def get_data():
    global conn
    while True:
        data = conn.recv(BUFFER_SIZE)
        data_np = process_data(data, i)
        print(data_np.shape)
        interval = seconds[-1] + 1
        seconds.append(interval)
        first = max(0, interval - 10)
        interval = max(10, interval)
        print(first, interval)
        X = np.linspace(first, interval, num=data_np.shape[0])
        Y = data_np
        return X, Y


@app.callback(Output("live-graph", "figure"), [Input("graph-update", "n_intervals")])
def update_graph_scatter(n):
    X, Y = get_data()
    data = plotly.graph_objs.Scatter(x=list(X), y=list(Y), name="Scatter", mode="lines")

    return {
        "data": [data],
        "layout": go.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[115, 180]),
        ),
    }


if __name__ == "__main__":
    app.run_server(debug=False, host="127.0.0.1", port="8080")
