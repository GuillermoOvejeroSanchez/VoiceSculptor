# Echo client program
import socket
import time


class Client:
    HOST = "127.0.0.1"  # The remote host
    PORT = 2345  # The same port as used by the server

    def __init__(self) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

    def connect(self):
        print("Connection")
        self.s.connect((self.HOST, self.PORT))

    def send(self, msg):
        self.s.sendall(bytes(msg, "utf-8"))

    def recv(self, size=1024):
        data = self.s.recv(size)
        return data

    def close(self):
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()


# Test
if __name__ == "__main__":
    c = Client()
    for i in range(10):
        msg = "Hello i={}".format(i)
        c.send(msg)
        time.sleep(0.2)
        # data = c.recv(1024)
        # print("Received", repr(data))
    c.close()