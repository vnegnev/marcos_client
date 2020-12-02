#!/usr/bin/env python3

import msgpack

version_major = 0
version_minor = 2
version_debug = 6
version_full = (version_major << 16) | (version_minor << 8) | version_debug

request_pkt = 0
emergency_stop_pkt = 1
close_server_pkt = 2
reply_pkt = 128

def construct_packet(data, packet_idx=0, command=request_pkt, version=(version_major, version_minor, version_debug)):
    vma, vmi, vd = version
    assert vma < 256 and vmi < 256 and vd < 256, "Version is too high for a byte!"
    version = (vma << 16) | (vmi << 8) | vd
    fields = [command, packet_idx, 0, version, data]
    return fields

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
            payload[4]['test_throughput']['array1'][-1], payload[4]['test_throughput']['array2'][-1]))
    except KeyError:
        print("Reply data: ")
        print(reply_data)

def send_packet(packet, socket):
    socket.sendall(msgpack.packb(packet))

    unpacker = msgpack.Unpacker()
    packet_done = False
    while not packet_done:
        buf = socket.recv(1024)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker: # ugly way of doing it
            return o # quit function after 1st reply (could make this a thread in the future)

def ba_flip_endian(ba):
    # Flip the endianness of the byte array, to suit the server hardware's strange convention
    N = len(ba)
    ba2 = bytearray(N)

    for k in range(N//4):
        ba2[4*k] = ba[4*k+3]
        ba2[4*k+1] = ba[4*k+2]
        ba2[4*k+2] = ba[4*k+1]
        ba2[4*k+3] = ba[4*k]

    return ba2
