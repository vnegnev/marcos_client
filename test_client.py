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

request = 0
emergency_stop = 1
close_server = 2

def construct_packet(data, packet_idx=0, command=request):
    assert version_major < 256 and version_minor < 256 and version_debug < 256, "Version is too high for a byte!"
    version = (version_major << 16) | (version_minor << 8) | version_debug
    fields = [command, packet_idx, 0, version, data]
    return msgpack.packb(fields)

def process(payload, print_all=False):
    # data = msgpack.unpackb(raw_reply, use_list=False, max_array_len=1024*1024)
    reply_data = payload[4]

    if print_all:
        print("")

        status = payload[5]

        try:
            print("Errors:")
            for k in status['errors']:
                print(k)
        except KeyError:
            pass

        try:
            print("Warnings:")
            for k in status['warnings']:
                print(k)
        except KeyError:
            pass

        try:
            print("Infos:")
            for k in status['infos']:
                print(k)
        except KeyError:
            pass

    try:
        print("Last elements of returned unsigned arrays: {:f}, {:f}".format(
            payload[4]['array1'][-1], payload[4]['array2'][-1]))
    except KeyError:
        print("Reply data: ")
        print(reply_data)

def send_msg(msg, socket):
    socket.sendall(msg)

    unpacker = msgpack.Unpacker()
    packet_done = False
    while not packet_done:
        buf = socket.recv(1024)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker: # ugly way of doing it
            return o # quit function after 1st reply (could make this a thread in the future)

def sock_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        for k in range(7):
            msg = construct_packet( {
                'test_length': 10**k,
                'test1':[1,2,3,4,5,6,7,8],
                'test2':{'oogabooga':[4,5,6,7]} })

            process(send_msg(msg, s))
            packet_idx += 2

        # Random other packet
        process(send_msg(construct_packet({'boo': 3}) , s))

        # Close server
        msg = construct_packet( {}, packet_idx, command=close_server)
        process(send_msg(msg, s), print_all=True)

    #print(repr(data))

if __name__ == "__main__":
    sock_test()
