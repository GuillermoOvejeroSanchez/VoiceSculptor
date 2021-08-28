#!/usr/bin/env python
import json
import os
import sys
import time
from datetime import datetime

import dash
import numpy as np
import pandas as pd
import pika
import plotly
import requests
from numpy import loads


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()

    channel.queue_declare(queue="sound_data")

    def callback(ch, method, properties, body):
        msg = json.loads(body)
        print(" [x] Received %r" % msg["npause"])

    channel.basic_consume(
        queue="sound_data", on_message_callback=callback, auto_ack=True
    )

    print(" [*] Waiting for messages. To exit press CTRL+C")
    msg = channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
