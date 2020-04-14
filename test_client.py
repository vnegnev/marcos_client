#!/usr/bin/env python3
import socket
import msgpack
import time

from local_config import ip_address, port

version_major = 0
version_minor = 0
version_debug = 1

def construct_packet(data, packet_idx=0):
    assert version_major < 256 and version_minor < 256 and version_debug < 256, "Version is too high for a byte!"
    version = (version_major << 16) | (version_minor << 8) & version_debug;
    fields = [0, packet_idx, 0, version, data]
    return msgpack.packb(fields, use_bin_type=True)
        

def sock_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        for k in range(100):
            msg = construct_packet( { 'test1':[1,2,3,4,5,6,7,8],
                                  'test2':{'oogabooga':[4,5,6,7]} }, packet_idx)
            time.sleep(1)
            # print(msg)
            s.sendall(msg)
            packet_idx += 1
            reply = s.recv(1024)
            print(reply)
        # for k in range(256):
        #     s.sendall(b"asdf")
        #data = s.recv(1000)

    #print(repr(data))

if __name__ == "__main__":
    sock_test()
