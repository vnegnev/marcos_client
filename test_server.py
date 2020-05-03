#!/usr/bin/env python3
import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt

import pdb
st = pdb.set_trace

from local_config import ip_address, port
from server_comms import *

class ServerTest(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    def setUp(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port))
        self.packet_idx = 0

    def tearDown(self):
        self.s.close()

    def test_version(self):
        packet = construct_packet({'asdfasdf':1}, self.packet_idx, version=(0,0,0))
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full, {'UNKNOWN1': -1},
                          {'errors': ['not all client commands were understood'],
                           'infos': ['Client version 0.0.0 differs slightly from server version 0.0.5']}])

    def test_bad_packet(self):
        packet = construct_packet([1,2,3])
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {},
                          {'errors': ['no commands present or incorrectly formatted request']}])
                         

    
    def test_throughput(self):
        packet = construct_packet({'test_throughput':10}, self.packet_idx)
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'test_throughput':
                           {'array1': [0.0, 1.01, 2.02, 3.0300000000000002, 4.04, 5.05, 6.0600000000000005, 7.07, 8.08, 9.09],
                            'array2': [10.1, 11.11, 12.120000000000001, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19]}}, {}]
        )

        for k in range(7):
            with self.subTest(i=k):
                ke = 10**k
                kf = ke - 1
                packet = construct_packet({'test_throughput': ke}, self.packet_idx)
                reply = send_packet(packet, self.s)
                self.assertAlmostEqual(reply[4]['test_throughput']['array1'][-1], 1.01 * kf)
                self.assertAlmostEqual(reply[4]['test_throughput']['array2'][-1], 1.01 * (kf+10) )

    def test_fpga_clk(self):
        packet = construct_packet({'fpga_clk': [0xdf0d, 0x03f03f30, 0x00100700]})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply, [reply_pkt, 1, 0, version_full, {'fpga_clk': 0}, {}])

    def test_fpga_clk_partial(self):
        packet = construct_packet({'fpga_clk': [0xdf0d,  0x03f03f30]})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'fpga_clk': -1},
                          {'errors': ["you only provided some FPGA clock control words; check you're providing all 3"]}]
        )

    def test_several_okay(self):
        packet = construct_packet({'rx_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 10, # 81.38ns sampling for 122.88 clock freq
                                   'rf_amp': 8000,
                                   'tx_samples': 40,
                                   'recomp_pul': True,
                                   'raw_tx_data': b"0123456789abcdef"*4096})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'rx_freq': 0, 'tx_div': 0, 'rf_amp': 0, 'tx_samples': 0, 'recomp_pul': 0, 'raw_tx_data': 0},
                          {'infos': [
                              'true RX freq: 13.440000 MHz',
                              'TX sample duration: 0.081380 us',
                              'true RF amp: 12.207218',
                              'tx data bytes copied: 65536']}]
        )

    def test_several_some_bad(self):
        packet = construct_packet({'rx_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 100000, # 813.8us sampling for 122.88 clock freq
                                   'rf_amp': 100, # TODO: make a test where this is too large
                                   'tx_samples': 40,
                                   'recomp_pul': False,
                                   'raw_tx_data': b"0123456789abcdef"*4097})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'rx_freq': 0, 'tx_div': -2, 'rf_amp': 0, 'tx_samples': 0, 'recomp_pul': -2, 'raw_tx_data': -1},
                          {'errors': ['too much raw TX data'],
                           'warnings': ['TX divider outside the range [1, 1000]; make sure this is what you want',
                                        'recomp_pul requested but set to false; doing nothing'],
                           'infos': ['true RX freq: 13.440000 MHz', 'TX sample duration: 813.802083 us', 'true RF amp: 0.152590']}])

    def test_gradient_offsets(self):
        commands = ( 'grad_offs_x', 'grad_offs_y', 'grad_offs_z', 'grad_offs_z2' )
        packet = construct_packet({commands[0]: -60000,
                                   commands[1]: 60000,
                                   commands[2]: 12345})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'grad_offs_x': 0, 'grad_offs_y': 0, 'grad_offs_z': 0},
                          {}])

    def test_acquire(self):
        samples = 1000000
        packet = construct_packet({'acq': samples})
        reply = send_packet(packet, self.s)
        acquired_data_raw = reply[4]['acq']
        data = np.frombuffer(acquired_data_raw, np.complex64)

        plt.plot(np.abs(data));plt.show()
        
        self.assertEqual(reply[:4], [reply_pkt, 1, 0, version_full])
        self.assertEqual(len(acquired_data_raw), samples*8)
        # st()

    @unittest.skip("rewrite needed")
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
