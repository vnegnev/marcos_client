#!/usr/bin/env python3
import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt

import pdb
st = pdb.set_trace

from local_config import ip_address, port
from server_comms import *

fpga_clk_freq_MHz = 122.88

class AcquireTest(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    def setUp(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port))
        self.packet_idx = 0

    def tearDown(self):
        self.s.close()

    def test_acquire(self):        
        # Top-level parameters
        rx_freq = 2 # LO frequency, MHz
        samples = 5000 # number of samples to acquire
        divisor = 2 # TX sampling rate divisor, i.e. 3 means TX samples are output at 1/3 of the rate of the FPGA clock
        # (minimum value is 2)
        rf_amp = 16384 # should be below 32768 if you're using both I/Q channels; otherwise below 65536

        # raw TX data generation
        tx_iq_freq = 1 # MHz        
        sample_period_us = 1/fpga_clk_freq_MHz # 1 / RP clock freq
        tx_bytes = 65536
        values = tx_bytes // 4
        xaxis = np.linspace(0, 1, values) * sample_period_us * divisor * values # in units of microseconds
        tx_arg = 2 * np.pi * xaxis * tx_iq_freq
        if False:
            # sinewaves
            tx_i = np.round(rf_amp * np.cos(tx_arg) ).astype(np.ushort) # note signed -> unsigned for binary maths!
            tx_q = np.round(rf_amp * np.sin(tx_arg) ).astype(np.ushort)
        else:
            # Gaussian envelope
            xmid = (xaxis[0] + xaxis[-1])/2
            gaus_sd = xmid/2
            tx_i = np.round(rf_amp * np.exp(-(xaxis - xmid) ** 2 / (gaus_sd ** 2) ) ).astype(np.ushort)
            # plt.plot(tx_i);plt.show()
            tx_q = tx_i
        
        raw_tx_data = bytearray(tx_bytes)
        for k, (si, sq) in enumerate(zip(tx_i, tx_q)):
            # TODO: do this assignment with Numpy slicing for performance
            raw_tx_data[4*k + 0] = si & 0xff
            raw_tx_data[4*k + 1] = si >> 8
            raw_tx_data[4*k + 2] = sq & 0xff
            raw_tx_data[4*k + 3] = sq >> 8
        
        # compute pulse sequence
        from ocra_lib.assembler import Assembler
        ass = Assembler()
        # sequence_byte_array = ass.assemble("ocra_lib/se_default_vn.txt") # no samples appear since no acquisition takes place
        sequence_byte_array = ass.assemble("ocra_lib/se_default_vn.txt")            
        
        packet = construct_packet({# 'rx_freq': 0x8000000,
            'rx_freq': int(np.round(rx_freq / fpga_clk_freq_MHz * (1 << 30))),
            'tx_div': divisor - 1,
            # 'rf_amp': rf_amp,
            # 'tx_samples': 1,
            # 'recomp_pul': True,
            'raw_tx_data': raw_tx_data,
            'seq_data': sequence_byte_array,
            'acq': samples})

        reply = send_packet(packet, self.s)
        for sti in ('infos','warnings','errors'):
            try:
                for k in reply[5][sti]:
                    print(k)
            except KeyError:
                pass

        if True:
            # receive some data
            packet = construct_packet({'acq': samples})
            for k in range(3):
                reply = send_packet(packet, self.s)
                acquired_data_raw = reply[4]['acq']
                data = np.frombuffer(acquired_data_raw, np.complex64)

            if False:
                mkr = '-' # '.'
                plt.plot(np.real(data), mkr);plt.show()
        
        self.assertEqual(reply[:4], [reply_pkt, 1, 0, version_full])
        self.assertEqual(len(acquired_data_raw), samples*8)
        # st()

def main_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        # throughput_test(s)
        test_client(s)
        # shutdown_server(s)
    
if __name__ == "__main__":
    # main_test()
    unittest.main()
