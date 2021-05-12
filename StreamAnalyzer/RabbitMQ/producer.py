#!/usr/bin/env python
import pika
from threading import Thread
import multiprocessing.dummy as mp
import time

connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
channel = connection.channel()

channel.queue_declare(queue="hello")

channel.basic_publish(exchange="", routing_key="hello", body="Hello World!")


def my_func(i):
    msg = "HelloMyFriend {}".format(i)
    channel.basic_publish(exchange="", routing_key="hello", body=msg)


total_msg = 10
# ini = time.time()
# for i in range(total_msg):
#     my_func(i)
# print(time.time() - ini)
# time.sleep(1)

ini = time.time()
p = mp.Pool(8)
p.map(my_func, range(0, 10000))  # range(0,1000) if you want to replicate your example
p.close()
p.join()
print(time.time() - ini)


connection.close()
