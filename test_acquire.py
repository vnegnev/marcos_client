#!/usr/bin/env python3
import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as sig

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
        rx_freq = 10 # LO frequency, MHz
        samples = 800 # number of samples to acquire
        divisor = 2 # TX sampling rate divisor, i.e. 3 means TX samples are output at 1/3 of the rate of the FPGA clock. Minimum value is 2
        sampling_divisor = 50 # as above, I think (+/- 1 error)
        
        sample_period_us = 1/fpga_clk_freq_MHz # 1 / RP clock freq
        rx_sample_period = sample_period_us * sampling_divisor * 2 # not sure where the 2x comes from
        
        rf_amp = 16384 # should be below 32768 if you're using both I/Q channels; otherwise below 65536

        # raw TX data generation
        tx_iq_freq = 0.195 # MHz
        tx_bytes = 65536
        values = tx_bytes // 4
        xaxis = np.linspace(0, 1, values) * sample_period_us * divisor * values # in units of microseconds
        tx_arg = 2 * np.pi * xaxis * tx_iq_freq
        if True:
            # sinewaves, for loopback testing
            tx_i = np.round(rf_amp * np.cos(tx_arg) ).astype(np.ushort) # note signed -> unsigned for binary maths!
            tx_q = np.round(rf_amp * np.sin(tx_arg) ).astype(np.ushort)
        else:
            # Gaussian envelope, for oscilloscope testing
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
            'rx_freq': int(np.round(rx_freq / fpga_clk_freq_MHz * (1 << 30))) & 0xfffffff0 | 0xf,
            'tx_div': divisor - 1,
            'rx_rate': sampling_divisor,
            'tx_size': 32767,
            #'rf_amp': rf_amp,
            #'tx_samples': 1,
            # 'recomp_pul': True,
            'raw_tx_data': raw_tx_data,
            'seq_data': sequence_byte_array,
            'acq': samples
        })

        reply = send_packet(packet, self.s)
        for sti in ('infos','warnings','errors'):
            try:
                for k in reply[5][sti]:
                    print(k)
            except KeyError:
                pass
        acquired_data_raw = reply[4]['acq']
        data = np.frombuffer(acquired_data_raw, np.complex64)
        # data = np.frombuffer(acquired_data_raw, np.uint64) # ONLY FOR DEBUGGING THE FIFO COUNT

        if True:
                # mkr = '-'
                mkr = '.'
                fig, axs = plt.subplots(2,1)
                axs[0].plot(data.real, '.')
                axs[0].plot(data.imag, '.')                

                # N = data.size
                # f_axis = fft.fftfreq(N, d=rx_sample_period)[:N//2]
                # spectrum = np.abs(fft.fft(sig.detrend(data)))[:N//2]
                f_axis, spectrum = sig.welch(data, fs=1/rx_sample_period, return_onesided=False)
                
                axs[1].plot(f_axis, spectrum, '.')
                print('max power at {:.3f} +/- {:.3f} MHz'.format(f_axis[np.argmax(spectrum)], f_axis[11]-f_axis[10]))
                for ax in axs:
                    ax.grid(True)
                plt.show()

        if False:
            # receive some data repeatedly after the server's been configured
            packet = construct_packet({'acq': samples})
            for k in range(3):
                reply = send_packet(packet, self.s)
                acquired_data_raw = reply[4]['acq']
                data = np.frombuffer(acquired_data_raw, np.complex64)        

        if False:
            self.assertEqual(reply[:4], [reply_pkt, 1, 0, version_full])
            self.assertEqual(len(acquired_data_raw), samples*8)

def main_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        # throughput_test(s)
        test_client(s)
        # shutdown_server(s)
    
if __name__ == "__main__":
    # main_test()
    unittest.main()
