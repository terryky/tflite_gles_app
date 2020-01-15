# coding: utf-8

import socket
import time
import datetime

PORT_NUM = 12345

class ClientSocket:

    def __init__(self):
        print ('Initialize client socket')

        self.client_skt =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_skt.connect(('127.0.0.1', PORT_NUM))
        self.client_skt.send("HELLO\0")
        data = self.client_skt.recv(1024)

    def run(self):

        count = 0
        while True:
            now_time = datetime.datetime.now()
            print('[C] {0}'.format(now_time.strftime('%Y/%m/%d %H:%M:%S')))

            self.client_skt.send('REQUEST{0}\0'.format(count))
            data = self.client_skt.recv(1024)

            count += 1
            time.sleep(1)

        return

if __name__ == "__main__":
    sock = ClientSocket()
    sock.run()

