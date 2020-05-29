#!/usr/bin/env python3
import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as sig

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz
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
        sequence_byte_array = ass.assemble("ocra_lib/se_default_vn.txt") # no samples appear since no acquisition takes place
        # sequence_byte_array = ass.assemble("ocra_lib/se_default.txt")        
        
        packet = construct_packet({# 'lo_freq': 0x8000000,
            'lo_freq': int(np.round(lo_freq / fpga_clk_freq_MHz * (1 << 30))) & 0xfffffff0 | 0xf,
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

        if False:
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
                # plt.show()

        if True:
            raw_grad_data = bytearray(4096 * 2)
            
            for k in range(100):
                # Ramp from min. to max. voltage
                ramp_samples = 90
                ramp = np.linspace(-1, 1, ramp_samples) # between -1 and 1, which are the DAC output full-scale limits (NOT voltage)
                ramp[0] = 1 # peak at the start

                # Sine wave
                sine_samples = 200
                cycles = 5
                sine_arg = np.linspace(0, 2*np.pi*cycles, sine_samples) 
                sine = np.sin(sine_arg) * np.exp(-sine_arg/10)

                # Concatenate
                dac_waveform = np.hstack([ramp, sine])
                
                # np.uint16: actually it needs to be a 16-bit 2's
                # complement signed int, but Python tracks the sign
                # independently from the value which complicates the
                # bitwise arithmetic in the for loop below
                dac_data = np.round(dac_waveform * 32767).astype(np.uint16)
                for k, r in enumerate(dac_data): # 8192//4):
                    assert k < 8192//4, "Too much data for the gradient RAM"
                    n = 4 * k
                    val = 0x00100000 | (r << 4) ;
                    raw_grad_data[n] = val & 0xff;
                    raw_grad_data[n+1] = (val >> 8) & 0xff;
                    raw_grad_data[n+2] = (val >> 16) & 0xff;
                    raw_grad_data[n+3] = (val >> 24) & 0xff;
                    
                    # raw_grad_data[n+4] = val2 & 0xff;
                    # raw_grad_data[n+5] = (val2 >> 8) & 0xff;
                    # raw_grad_data[n+6] = (val2 >> 16) & 0xff;
                    # raw_grad_data[n+7] = (val2 >> 24) & 0xff;

                # # OLD, but needed to initialise the DACs somehow
                if True:
                    val = 0x00100000 | (((k * 655 - 32768) & 0xffff) << 4) ;
                    val2 = 0x00200002;

                    raw_grad_data[0] = val & 0xff;
                    raw_grad_data[1] = (val >> 8) & 0xff;
                    raw_grad_data[2] = (val >> 16) & 0xff;
                    raw_grad_data[3] = (val >> 24) & 0xff;

                    raw_grad_data[4] = val2 & 0xff;
                    raw_grad_data[5] = (val2 >> 8) & 0xff;
                    raw_grad_data[6] = (val2 >> 16) & 0xff;
                    raw_grad_data[7] = (val2 >> 24) & 0xff;                    

                # number of samples acquired will determine how long the actual sequence runs for, both RF and gradients,
                # since the processor is put into the reset state by the server once acquisition is complete
                packet = construct_packet({'acq': 5000, 
                                           'grad_mem_x': raw_grad_data,
                                           })
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
                
        if False:
            # ramp the x gradient voltage offset
            for k in range(100):
                packet = construct_packet({'acq': 100,
                                           'grad_offs_x': (k * 655 - 32768) & 0xffff}) # .astype(np.int16)})
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
