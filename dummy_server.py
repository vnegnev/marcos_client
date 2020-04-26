#!/usr/bin/env python3

## THIS DOESN'T DO ANYTHING USEFUL YET

import socket
import msgpack

def sock_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # s.connect( ("localhost", 11111) )
        s.bind( ("localhost", 11111) )
        s.listen()
        # msg = msgpack.packb({'sdf':3, 'asdf':4})
        conn, addr = s.accept()
        with conn:
            print("Connected by ", addr)
            data = conn.recv(1024)
            print(data)

if __name__ == "__main__":
    sock_test()
