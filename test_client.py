#!/usr/bin/env python3
import socket
import msgpack
import time

import pdb
st = pdb.set_trace

from local_config import ip_address, port

version_major = 0
version_minor = 0
version_debug = 1

def construct_packet(data, packet_idx=0, emergency_stop=False):
    assert version_major < 256 and version_minor < 256 and version_debug < 256, "Version is too high for a byte!"
    version = (version_major << 16) | (version_minor << 8) | version_debug
    fields = [int(emergency_stop), packet_idx, 0, version, data]
    return msgpack.packb(fields)

def process(data, print_all=False):
    # data = msgpack.unpackb(raw_reply, use_list=False, max_array_len=1024*1024)

    if print_all:
        print("")

        try:
            print("Errors:")
            for k in data[5]['errors']:
                print(k)
        except KeyError:
            pass

        try:
            print("Warnings:")
            for k in data[5]['warnings']:
                print(k)
        except KeyError:
            pass

        try:
            print("Infos:")
            for k in data[5]['infos']:
                print(k)
        except KeyError:
            pass

    print("Last elements of returned unsigned arrays: {:f}, {:f}".format(
        data[4]['data'][-1], data[4]['data2'][-1]))

def sock_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        for k in range(2):
            msg = construct_packet( { 'test1':[1,2,3,4,5,6,7,8],
                                      'test2':{'oogabooga':[4,5,6,7]} }, packet_idx, emergency_stop=False)
            # time.sleep(1)
            # print(msg)
            s.sendall(msg)
            packet_idx += 2

            unpacker = msgpack.Unpacker()
            packet_done = False
            while not packet_done:
                buf = s.recv(1024)
                if not buf:
                    break
                unpacker.feed(buf)
                for o in unpacker: # ugly way of doing it
                    process(o)
                    packet_done = True
                    break
                    
        # for k in range(256):
        #     s.sendall(b"asdf")
        #data = s.recv(1000)

    #print(repr(data))

if __name__ == "__main__":
    sock_test()
