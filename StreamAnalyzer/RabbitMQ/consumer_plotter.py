#!/usr/bin/env python
from numpy import loads
import pika, sys, os
import pandas as pd
import json
import plotly
import requests
from datetime import datetime
import numpy as np
import time
import dash


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