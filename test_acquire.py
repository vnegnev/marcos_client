#!/usr/bin/env python3
import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as sig

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
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
        # Top-level parameters
        lo_freq = 2 # LO frequency, MHz
        samples = 800 # number of samples to acquire
        divisor = 2 # TX sampling rate divisor, i.e. 3 means TX samples are output at 1/3 of the rate of the FPGA clock. Minimum value is 2
        sampling_divisor = 123 # as above, I think (+/- 1 error); 123 means it'll be around 1 sample per us
        
        sample_period_us = 1/fpga_clk_freq_MHz # 1 / RP clock freq
        rx_sample_period = sample_period_us * sampling_divisor * 2 # not sure where the 2x comes from
        
        rf_amp = 16384 # should be below 32768 if you're using both I/Q channels; otherwise below 65536

        # raw TX data generation
        tx_iq_freq = 0.195 # MHz
        tx_bytes = 65536
        values = tx_bytes // 4
        xaxis = np.linspace(0, 1, values) * sample_period_us * divisor * values # in units of microseconds
        tx_arg = 2 * np.pi * xaxis * tx_iq_freq
        if False:
            # sinewaves, for loopback testing
            tx_i = np.round(rf_amp * np.cos(tx_arg) ).astype(np.uint16) # note signed -> unsigned for binary maths!
            tx_q = np.round(rf_amp * np.sin(tx_arg) ).astype(np.uint16)
        else:
            # Gaussian envelope, for oscilloscope testing
            xmid = (xaxis[0] + xaxis[-1])/2
            gaus_sd = xmid/2
            tx_i = np.round(rf_amp * np.exp(-(xaxis - xmid) ** 2 / (gaus_sd ** 2) ) ).astype(np.uint16)
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
        sequence_byte_array = ass.assemble("ocra_lib/gpa_test.txt") # no samples appear since no acquisition takes place
        # sequence_byte_array = ass.assemble("ocra_lib/se_default.txt")        
        
        packet = construct_packet({# 'lo_freq': 0x8000000,
            'lo_freq': int(np.round(lo_freq / fpga_clk_freq_MHz * (1 << 30))) & 0xfffffff0 | 0xf,
            'tx_div': divisor - 1,
            'rx_div': sampling_divisor,
            'tx_size': 32767,
            'raw_tx_data': raw_tx_data,
            # 'acq': samples,
            'seq_data': sequence_byte_array
        })

        reply = send_packet(packet, self.s)
        for sti in ('infos','warnings','errors'):
            try:
                for k in reply[5][sti]:
                    print(k)
            except KeyError:
                pass

        if True:
            raw_grad_data_x = bytearray(8192)
            
            for k in range(100):
                # Ramp from min. to max. voltage
                ramp_samples = 100
                ramp = np.linspace(-1, 1, ramp_samples) # between -1 and 1, which are the DAC output full-scale limits (NOT voltage)

                # Sine wave
                sine_samples = 203
                cycles = 5
                sine_arg = np.linspace(0, 2*np.pi*cycles, sine_samples) 
                sine = np.sin(sine_arg) * np.exp(-sine_arg/10)

                # Concatenate
                dac_waveform = np.hstack([ramp, sine])
                dac_waveform[0] = 0
                dac_waveform[1] = 1
                dac_waveform[299] = 0
                dac_waveform[300] = 0
                dac_waveform[301] = 1
                dac_waveform[302] = 0                
                
                # np.uint16: actually it needs to be a 16-bit 2's
                # complement signed int, but Python tracks the sign
                # independently from the value which complicates the
                # bitwise arithmetic in the for loop below
                dac_data = np.round(dac_waveform * 32767).astype(np.uint16)

                ## ocra1 data
                if grad_board == "ocra1":
                    grad_core_select = 0x1
                    
                    raw_gd = np.zeros(2048, dtype=np.uint32)
                    raw_gd[:dac_data.size] = (dac_data.astype(np.uint32) << 4) | 0x00100000

                    # direct way of doing the init (only 1st element)
                    raw_gd[0] = 0x00200002
                    # raw_gd[1] = 0x00100000 | (((k * 655 - 32768) & 0xffff) << 4) # slowly-rising peak as the outer loop continues
                    # raw_gd[1], raw_gd[2], raw_gd[3] = raw_gd[0], raw_gd[0], raw_gd[0]
                    # raw_gd[5], raw_gd[6], raw_gd[7] = raw_gd[4], raw_gd[4], raw_gd[4]

                    ## Extend X data to Y, Z, Z2
                    raw_grad_data = np.empty(8192, dtype=np.uint32)
                    raw_grad_data[0::4] = raw_gd # channel 0
                    raw_grad_data[1::4] = (raw_gd | (1 << 25) ) # channel 1
                    raw_grad_data[2::4] = (raw_gd | (2 << 25) ) # channel 2
                    raw_grad_data[3::4] = (raw_gd | (3 << 25) | (1 << 24) ) # channel 3 and broadcast

                elif grad_board == "gpa-fhdo":
                    grad_core_select = 0x2
                    
                    raw_gd = np.ones(2048, dtype=np.uint32) * 0x8000
                    # raw_gd[0] = 0x00000f
                    # raw_gd[0] = 0x030100 # init
                    # raw_gd[1] = 0x020000 # sync
                    raw_gd[0:dac_data.size] = (dac_data + (1 << 15)).astype(np.uint32)
                    
                    # raw_gd[0] = 0x111111 # init
                    # raw_gd[0] = 0x000000
                    # raw_gd[1:8] = [0x2000, 0x4000, 0x6000, 0xffff, 0x5000, 0x4000, 0x0000]
                    # raw_gd[8:] = np.tile(raw_gd[0:8], 255)

                    ## Extend X data to Y, Z, Z2
                    raw_grad_data = np.empty(8192, dtype=np.uint32)
                    raw_grad_data[0::4] = raw_gd | (1 << 19) # channel 0
                    raw_grad_data[1::4] = raw_gd | (1 << 19) | (1 << 25) | (1 << 16) # channel 1
                    raw_grad_data[2::4] = raw_gd | (1 << 19) | (2 << 25) | (2 << 16) # channel 2
                    raw_grad_data[3::4] = raw_gd | (1 << 19) | (3 << 25) | (1 << 24) | (3 << 16) # channel 3 and broadcast
                    # raw_grad_data[3::4] = np.ones_like(raw_gd, dtype=np.uint32) * 0x020000 # sync before ch0 transmissions

                # number of samples acquired will determine how long the actual sequence runs for, both RF and gradients,
                # since the processor is put into the reset state by the server once acquisition is complete
                packet = construct_packet({'acq': 520,
                                           'grad_mem': raw_grad_data.tobytes(),
                                           'grad_ser': grad_core_select,
                                           # 'grad_div': (1022, 63), # slowest rate, for basic testing
                                           # 'grad_div': (1022, 6) # slowest interval, fastest working SPI on gpa-fhdo board with 1.5m Ethernet cable
                                           'grad_div': (250, 8)
                                           # 'grad_div': (189, 6) # fastest sustained 4-channel update settings on gpa-fhdo
                                           # 'grad_div': (150, 1)
                                           # 'grad_div': (107, 10)
                                           # 'grad_div': (9, 1) # fastest sustained 4-channel update settings on ocra1
                                           })
                reply = send_packet(packet, self.s)

                acquired_data_raw = reply[4]['acq']
                data = np.frombuffer(acquired_data_raw, np.complex64)
                return_status = reply[5]

                for sti in ('infos','warnings','errors'):
                    try:
                        for k in return_status[sti]:
                            print(k)
                    except KeyError:
                        pass

                assert 'errors' not in reply[5], reply[5]['errors'][0]
                    

                if False and k == 1: # plot acquired data
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
            # ramp the x gradient voltage offset
            for k in range(100):
                grad_offs = (k * 655 - 32768) & 0xffff # .astype(np.int16)})
                packet = construct_packet({'acq': 100,
                                           'grad_offs_x': grad_offs,
                                           'grad_offs_y': grad_offs,
                                           'grad_offs_z': grad_offs})
                reply = send_packet(packet, self.s)

                acquired_data_raw = reply[4]['acq']
                data = np.frombuffer(acquired_data_raw, np.complex64)
                # time.sleep(0.1)

                for sti in ('infos','warnings','errors'):
                    try:
                        for k in reply[5][sti]:
                            print(k)
                    except KeyError:
                        pass
                
            # for k in range(3):
            #     reply = send_packet(packet, self.s)
            #     acquired_data_raw = reply[4]['acq']
            #     data = np.frombuffer(acquired_data_raw, np.complex64)        

        # acquired_data_raw = reply[4]['acq']
        # data = np.frombuffer(acquired_data_raw, np.complex64)
        # data = np.frombuffer(acquired_data_raw, np.uint64) # ONLY FOR DEBUGGING THE FIFO 
            
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
