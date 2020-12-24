#!/usr/bin/env python3
#
# To run a single test, use e.g.:
# python -m unittest test_server.ServerTest.test_bad_packet

import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt
import warnings

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
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
        versions = [ (0,2,0), (0,2,5), (0,3,100), (0,3,255), (1,5,7), (255,255,255) ]

        def diff_info(client_ver):
            return {'infos': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' differs slightly from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug)],
             'errors': ['not all client commands were understood']}

        def diff_warning(client_ver):
            return {'warnings': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' different from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug)],
             'errors': ['not all client commands were understood']}

        def diff_error(client_ver):
            return {'errors': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' significantly different from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug),
                               'not all client commands were understood']}
        
        # results = 
        expected_outcomes = [diff_info, diff_info, diff_warning, diff_warning, diff_error, diff_error]
        
        for v, ee in zip(versions, expected_outcomes):
            packet = construct_packet({'asdfasdf':1}, self.packet_idx, version=v)
            reply = send_packet(packet, self.s)
            self.assertEqual(reply,
                             [reply_pkt, 1, 0, version_full, {'UNKNOWN1': -1},
                              ee(v)])

    def test_bad_packet(self):
        packet = construct_packet([1,2,3])
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {},
                          {'errors': ['no commands present or incorrectly formatted request']}])
                         

    
    def test_bus(self):
        loops = 1000000
        packet = construct_packet({'test_bus':loops}, self.packet_idx)
        reply = send_packet(packet, self.s)
        null_t, read_t, write_t = reply[4]['test_bus']

        # print("{:d}, {:d}, {:d}".format(null_t, read_t, write_t))
        self.assertAlmostEqual(null_t, 1, delta=1) # 1 million numerical operations take ~1 us (might be getting optimised out)
        loops_norm = loops/1e6
        self.assertAlmostEqual(read_t/1e3, 141.9 * loops_norm, delta = 2 * loops_norm) # 1 read takes ~141.9 ns on average
        self.assertAlmostEqual(write_t/1e3, 157.9 * loops_norm, delta = 2 * loops_norm) # 1 write takes ~157.9 ns on average

    def test_io(self):
        packet = construct_packet({'test_net':10}, self.packet_idx)

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
        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 10,
                                   'rx_div': 250,
                                   'tx_size': 32767,
                                   'raw_tx_data': b"0000000000000000"*4096,
                                   'grad_div': (303, 32),
                                   'grad_ser': 1,
                                   'grad_mem': b"0000"*8192,
                                   'acq_rlim':10000,
                                   })
        reply = send_packet(packet, self.s)
        
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': 0, 'rx_div': 0,
                           'tx_size': 0, 'raw_tx_data': 0, 'grad_div': 0, 'grad_ser': 0,
                           'grad_mem': 0, 'acq_rlim': 0},
                          {'infos': [
                              'tx data bytes copied: 65536',
                              'gradient mem data bytes copied: 32768']}]
        )

    def test_several_some_bad(self):
        # first, send a normal packet to ensure everything's in a known state
        packetp = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                    'tx_div': 10, # 81.38ns sampling for 122.88 clock freq, 80ns for 125
                                    'rx_div': 250,
                                    'raw_tx_data': b"0000000000000000"*4096
        })
        send_packet(packetp, self.s)

        # Now, try sending with some issues
        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 100000,
                                   'rx_div': 32767,
                                   'tx_size': 65535,
                                   'raw_tx_data': b"0123456789abcdef"*4097,
                                   'grad_div': (1024, 0),
                                   'grad_ser': 16,
                                   'grad_mem': b"0000"*8193,
                                   'acq_rlim': 10,
                                   })
        
        reply = send_packet(packet, self.s)
        
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': -1, 'rx_div': -1, 'tx_size': -1, 'raw_tx_data': -1, 'grad_div': -1, 'grad_ser': -1, 'grad_mem': -1, 'acq_rlim': -1},
                          {'errors': ['TX divider outside the range [1, 10000]; check your settings',
                                      'RX divider outside the range [25, 8192]; check your settings',
                                      'TX size outside the range [1, 32767]; check your settings',
                                      'too much raw TX data',
                                      'grad SPI clock divider outside the range [1, 63]; check your settings',
                                      'serialiser enables outside the range [0, 0xf], check your settings',
                                      'too much grad mem data: 32772 bytes > 32768',
                                      'acquisition retry limit outside the range [1000, 10,000,000]; check your settings'
                                      ]}
                          ])

    def test_grad_adc(self):
        if grad_board != "gpa-fhdo":
            return

        # set up SPI to deliberately be too fast for the ADC -- latest server should automatically compensate
        send_packet(construct_packet({'grad_div': (200, 5), 'grad_ser': 2}), self.s)

        # # ADC defaults
        words = [ 0x40850000, # ADC reset
                  0x400b0600, 0x400d0600, 0x400f0600, 0x40110600, # input ranges for each ADC channel
                 ] # TODO: set outputs to ~0

        expected = [ 0xffff, 0x0600, 0x0600, 0x0600, 0x0600 ]
        readback = []
        
        for w in words:
            send_packet(construct_packet({'grad_dir': w}), self.s)
            readback.append( send_packet(construct_packet({'grad_adc': 1}), self.s)[4]['grad_adc'] )
            # if readback != r:
            #     warnings.warn( "ADC data expected: 0x{:0x}, observed 0x{:0x}".format(w, readback) )

        self.assertEqual(expected, readback)

    def test_state(self):        
        # Check will behave differently depending on the STEMlab version we're connecting to (and its clock frequency)
        true_rx_freq = '13.440000' if fpga_clk_freq_MHz == 122.88 else '13.671875'
        tx_sample_duration = '0.081380' if fpga_clk_freq_MHz == 122.88 else '0.080000'
        rx_sample_duration = 0

        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 10,
                                   'rx_div': 250,                                   
                                   'grad_div': (303, 32),
                                   'state': fpga_clk_freq_MHz * 1e6
                                   })
        reply = send_packet(packet, self.s)

        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': 0, 'rx_div': 0, 'grad_div': 0,
                           'state': 0},
                          {'infos': [
                              'LO frequency [CHECK]: 13.440000 MHz',
                              'TX sample duration [CHECK]: 0.070000 us',
                              'RX sample duration [CHECK]: 2.034505 us',
                              'gradient sample duration (*not* DAC sampling rate): 2.149000 us',
                              'gradient SPI transmission duration: 5.558000 us']}])
        
    def test_gradient_mem(self):
        grad_mem_bytes = 4 * 8192

        # everything should be fine
        raw_data = bytearray(grad_mem_bytes)
        for m in range(grad_mem_bytes):
            raw_data[m] = m & 0xff
        packet = construct_packet({'grad_mem' : raw_data})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'grad_mem': 0},
                          {'infos': ['gradient mem data bytes copied: {:d}'.format(grad_mem_bytes)] }
                          ])

        # a bit too much data
        raw_data = bytearray(grad_mem_bytes + 1)
        for m in range(grad_mem_bytes):
            raw_data[m] = m & 0xff
        packet = construct_packet({'grad_mem' : raw_data})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'grad_mem': -1},
                          {'errors': ['too much grad mem data: {:d} bytes > {:d}'.format(grad_mem_bytes + 1, grad_mem_bytes)] }
                          ])        

    def test_acquire_simple(self):
        # For comprehensive tests, see test_loopback.py
        samples = 10
        packet = construct_packet({'acq': samples})
        reply = send_packet(packet, self.s)
        acquired_data_raw = reply[4]['acq']
        data = np.frombuffer(acquired_data_raw, np.complex64)

        self.assertEqual(reply[:4], [reply_pkt, 1, 0, version_full])
        self.assertEqual(len(acquired_data_raw), samples*8)        
        self.assertIs(type(data), np.ndarray)
        self.assertEqual(data.size, samples)

        if False:
            plt.plot(np.abs(data));plt.show()

    @unittest.skip("rewrite needed")
    def test_bad_packet_format(self):
        packet = construct_packet({'configure_hw':
                                   {'lo_freq': 7.12345, # floats instead of uints
                                    'tx_div': 1.234}})
        reply_packet = send_packet(packet, self.s)
        # CONTINUE HERE: this should be handled gracefully by the server
        st()
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full, {'configure_hw': 3}, {}]
        )

    @unittest.skip("comment this line out to shut down the server after testing")
    def test_zzz_exit(self): # last in alphabetical order
        packet = construct_packet( {}, 0, command=close_server_pkt)
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full, {}, {'infos': ['Shutting down server.']}])

def throughput_test(s):
    packet_idx = 0

    for k in range(7):
        msg = msgpack.packb(construct_packet({'test_server_throughput': 10**k}))

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
