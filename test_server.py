#!/usr/bin/env python3
import socket, time, unittest
import msgpack

import pdb
st = pdb.set_trace

from local_config import ip_address, port

version_major = 0
version_minor = 0
version_debug = 4
version_full = (version_major << 16) | (version_minor << 8) | version_debug

request = 0
emergency_stop = 1
close_server = 2
reply = 128

def construct_packet(data, packet_idx=0, command=request, version=(version_major, version_minor, version_debug)):
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

class ServerTest(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    def setUp(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port))
        self.packet_idx = 0

    def tearDown(self):
        self.s.close()

    # def setUp(self):
    #     pass
    def test_version(self):
        packet = construct_packet({'asdfasdf':1}, self.packet_idx, version=(0,0,0))
        reply_packet = send_packet(packet, self.s)
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full, {'UNKNOWN1': 0},
                          {'errors': ['not all client commands were understood'],
                           'infos': ['Client version 0.0.0 differs slightly from server version 0.0.4']}])

    def test_throughput(self):
        packet = construct_packet({'test_throughput':10}, self.packet_idx)
        reply_packet = send_packet(packet, self.s)
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full,
                          {'test_throughput':
                           {'array1': [0.0, 1.01, 2.02, 3.0300000000000002, 4.04, 5.05, 6.0600000000000005, 7.07, 8.08, 9.09],
                            'array2': [10.1, 11.11, 12.120000000000001, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19]}}, {}]
        )

    def test_configure_hw_fpga_clk(self):
        packet = construct_packet({'configure_hw':
                                   {'fpga_clk': [0xdf0d, 0x03f03f30, 0x00100700]}})
        reply_packet = send_packet(packet, self.s)
        self.assertEqual(reply_packet, [reply, 1, 0, version_full, {'configure_hw': 1}, {}])

    def test_configure_hw_fpga_clk_partial(self):
        packet = construct_packet({'configure_hw':
                                   {'fpga_clk': [0xdf0d,  0x03f03f30]}})
        reply_packet = send_packet(packet, self.s)
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full,
                          {'configure_hw': 0},
                          {'errors': ["you only provided some FPGA clock control words; check you're providing all 3", 'not all configure_hw commands were successfully executed']}]
        )

    def test_configure_hw_several(self):
        packet = construct_packet({'configure_hw':
                                   {'rx_freq': 0x7000000, # floats instead of uints
                                    'tx_div': 10, # 81.38ns sampling for 122.88 clock freq
                                    'rf_amp': 8000,
                                    'tx_samples': 40,
                                    'recomp_pul': True,
                                    'raw_tx_data': b"0123456789abcdef"*4096}})
        reply_packet = send_packet(packet, self.s)
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full,
                          {'configure_hw': 6},
                          {'infos': [
                              'true RX freq: 13.440000 MHz',
                              'TX sample duration: 0.081380 us',
                              'true RF amp: 12.207218',
                              'tx data bytes copied: 65536']}]
        )

    def test_bad_packet_format(self):
        packet = construct_packet({'configure_hw':
                                   {'rx_freq': 7.12345, # floats instead of uints
                                    'tx_div': 1.234}})
        reply_packet = send_packet(packet, self.s)
        # CONTINUE HERE: this should be handled gracefully by the server
        st()
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full, {'configure_hw': 3}, {}]
        )

def throughput_test(s):
    packet_idx = 0

    for k in range(7):
        msg = msgpack.packb(construct_packet({'test_throughput': 10**k}))

        process(send_msg(msg, s))
        packet_idx += 2

def random_test(s):
    # Random other packet
    process(send_msg(msgpack.packb(construct_packet({'boo': 3}) , s)))
        
def shutdown_server(s):
    msg = msgpack.packb(construct_packet( {}, 0, command=close_server))
    process(send_msg(msg, s), print_all=True)    
    
def test_client(s):
    packet_idx = 0
    pkt = construct_packet( {
        'configure_hw': {
            'fpga_clk_word1': 0x1,
            'fpga_clk_word2': 0x2
            # 'fpga_clk_word3': 0x3,
        },
    }, packet_idx)
    process(send_msg(msgpack.packb(pkt), s), print_all=True)

def main_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        # throughput_test(s)
        test_client(s)
        # shutdown_server(s)
    
if __name__ == "__main__":
    # main_test()
    unittest.main()
