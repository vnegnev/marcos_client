#!/usr/bin/env python3
import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt

import pdb
st = pdb.set_trace

from local_config import ip_address, port
from server_comms import *

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
        # send pulse sequence
        from ocra_lib.assembler import Assembler
        ass = Assembler()
        sequence_byte_array = ass.assemble("ocra_lib/se_default.txt")
        
        # set acquisition rate and send sequence
        packet = construct_packet({'rx_freq': 0x1000000,
                                   'tx_div': 1000,
                                   'rf_amp': 000,
                                   'tx_samples': 400,
                                   'recomp_pul': True,
                                   'raw_tx_data': ba_flip_endian(sequence_byte_array)})
                                   # 'raw_tx_data': sequence_byte_array})
        reply = send_packet(packet, self.s)
        for k in reply[5]['infos']:
            print(k)

        # receive some data
        samples = 5000
        packet = construct_packet({'acq': samples})
        reply = send_packet(packet, self.s)
        acquired_data_raw = reply[4]['acq']
        data = np.frombuffer(acquired_data_raw, np.complex64)

        plt.plot(np.real(data), '.');plt.show()
        
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
