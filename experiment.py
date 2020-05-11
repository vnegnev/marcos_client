#!/usr/bin/env python3
# 
# Basic toolbox for server operations; wraps up a lot of stuff to avoid the need for hardcoding on the user's side.

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as sig

import pdb
st = pdb.set_trace

from local_config import ip_address, port
from server_comms import *

class Experiment:
    """ Wrapper class for managing an entire experimental sequence 
    samples: number of (I,Q) samples to acquire during a shot of the experiment
    lo_freq: local oscillator frequency, MHz
    tx_t: RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (for the STEMlab-122, it's 122.88 MHz). For example if tx_t = 1000, then a new RF TX sample will be output approximately every microsecond.
    (self.tx_t will have the true value after construction.)
    rx_t: RF RX sampling time in microseconds; as above (approximately). If samples = 100 and rx_t = 1.5, then samples will be taken for 150 us total.    
    """

    def __init__(self,
                 samples=1000,
                 lo_freq=5,
                 tx_t=0.1,
                 rx_t=0.5):
        self.samples = samples

        self.lo_freq_bin = int(np.round(lo_freq / fpga_clock_freq_MHz * (1 << 30))) & 0xfffffff0 | 0xf
        self.lo_freq = self.lo_freq_bin * fpga_clock_freq_MHz / (1 << 30)
                
        self.rx_div = int(np.round(rx_t * fpga_clock_freq_MHz))
        self.rx_t = self.rx_div / fpga_clock_freq_MHz
        
        self.tx_div = int(np.round(tx_t * fpga_clock_freq_MHz))
        self.tx_t = self.tx_div / fpga_clock_freq_MHz

        # Segments for RF TX and gradient BRAMs
        self.tx_offsets = []
        self.current_tx_offset = 0

        self.grad_offsets = []
        self.current_grad_offset = 0

    def add_tx(self, vec):
        """ vec: complex vector in the I,Q range [-1,1] and [-j,j]; units of full-scale RF DAC output.
        (Note that the magnitude of each element must be <= 1, i.e. np.abs(1+1j) is sqrt(2) and thus too high.)
        
        Returns the index of the relevant vector, which can be used later when the pulse sequence is being compiled.
        """                
        self.tx_offsets.append(self.current_tx_offset)
        self.current_tx_offset += vec.size
        try:
            self.tx_data = np.hstack( [self.tx_data, vec] )
        except AttributeError:
            self.tx_data = vec

        return len(self.tx_offsets) - 1

    def add_grad_x(self, vec):
        """ vec: real vector in the range [-1,1] units of full-scale gradient DAC output.
        
        Returns the index of the relevant vector, which can be used later when the pulse sequence is being compiled.
        """
        self.grad_offsets.append(self.current_grad_offset)
        self.current_grad_offset += vec.size
        try:
            self.grad_data = np.hstack( [self.grad_data, vec] )
        except AttributeError:
            self.grad_data = vec

        return len(self.grad_offsets) - 1

    def compile_tx_data(self):
        """ go through the TX data and prepare binary array to send to the server """
        self.tx_bytes = bytearray(self.tx_data.size * 4)
        if np.any(np.abs(self.tx_data) > 1.0):
            warnings.warn("TX data too large! Overflow will occur.")
        
        tx_i = np.round(32767 * self.tx_data.real).astype(np.uint16)
        tx_q = np.round(32767 * self.tx_data.imag).astype(np.uint16)

        self.tx_bytes[::4] = tx_i & 0xff
        self.tx_bytes[1::4] = tx_i >> 8
        self.tx_bytes[2::4] = tx_q & 0xff
        self.tx_bytes[3::4] = tx_q >> 8

    def compile_grad_data(self):
        """ go through the grad X data and prepare binary array to send to the server """
        self.grad_x_bytes = bytearray(self.grad_data.size * 4)
        if np.any(np.abs(self.grad_data) > 1.0):
            warnings.warn("Grad data too large! Overflow will occur.")
        
        gr = np.round(32767 * self.grad_data).astype(np.uint16)
        
        # TODO: check that this makes sense relative to test_acquire
        self.grad_x_bytes[::4] = (gr & 0xf) << 4
        self.grad_x_bytes[1::4] = (gr & 0xff0) >> 4
        self.grad_x_bytes[2::4] = (gr >> 12) | 0x10
        self.grad_x_bytes[3::4] = 0 # wasted?

    def run(self):
        """ compile the TX and grad data, send everything over.
        Returns the resultant data """
        self.compile_data()


def test_Experiment():
    exp = Experiment()
    
    # first TX segment
    t = np.linspace(0, 100, 1001) # goes to 100us, samples every 100ns
    freq = 0.2 # MHz
    tx_x = np.cos(2*np.pi*freq*t) + 1j*np.sin(2*np.pi*freq*t)
    tx_idx = exp.add_tx(tx_x)

    # first gradient segment
    tg = np.linspace(0, 50, 51) # goes to 50us, samples every 1us (sampling rate is fixed right now)
    tmean = 25
    tstd = 10

    grad = np.exp(-(tg-tmean)**2/tstd**2) # Gaussian 
    grad_idx = exp.add_grad_x(grad)    

    data = exp.run()
    
    # plt.plot(tg, grad)
    # plt.show()
        
if __name__ == "__main__":
    test_Experiment()
    
