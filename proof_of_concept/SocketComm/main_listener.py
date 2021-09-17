# Echo server program
import select
import socket
from json import loads

import numpy as np

# ---- Configuration ----
HOST = "127.0.0.1"  # Symbolic name meaning all available interfaces
PORT = 2345  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
readable = [s]  # list of readable sockets. s is readable if a client is waiting.
i = 0
# -----------------------

nsyll: int = 0


def process_data(data: bytes, i: int):
    global nsyll
    data = np.frombuffer(data)
    print(data)
    # data_string: str = data.decode("utf-8")
    # data_dict: dict = loads(data_string)
    # if i % 10 == 0:
    #     nsyll += data_dict["nsyll"]
    # print(data_dict)
    # print(nsyll)


# ---- Listener ----
# TODO Make GUI here
while True:
    # r will be a list of sockets with readable data
    r, w, e = select.select(readable, [], [], 0)
    rs: socket.socket
    for rs in r:  # iterate through readable sockets
        print((type(rs)))
        if rs is s:  # is it the server
            c, a = s.accept()
            print(c, a)
            print("\r{}:".format(a), "connected")
            readable.append(c)  # add the client
        else:
            # read from a client
            data = rs.recv(1024)
            if not data:
                print("\r{}:".format(rs.getpeername()), "disconnected")
                readable.remove(rs)
                rs.close()
            else:
                process_data(data, i)
                # print("\r{}:".format(rs.getpeername()), data)
    # a simple spinner to show activity
    i += 1
    print("/-\|"[i % 4] + "\r", end="", flush=True)
