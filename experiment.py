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

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
from ocra_lib.assembler import Assembler
import server_comms as sc

class Experiment:
    """Wrapper class for managing an entire experimental sequence 

    samples: number of (I,Q) samples to acquire during a shot of the experiment

    lo_freq: local oscillator frequency, MHz

    tx_t: RF TX sampling time in microseconds; will be rounded to a
    multiple of system clocks (for the STEMlab-122, it's 122.88
    MHz). For example if tx_t = 1000, then a new RF TX sample will be
    output approximately every microsecond.  (self.tx_t will have the
    true value after construction.)

    rx_t: RF RX sampling time in microseconds; as above
    (approximately). If samples = 100 and rx_t = 1.5, then samples will be
    taken for 150 us total.  grad

    instruction_file: path to an assembly text file, which will be
    compiled by the OCRA assembler.py.  If this is not supplied, the
    instruction bytecode should be supplied manually using
    define_instructions() before run() is called.

    OPTIONAL PARAMETERS - ONLY ALTER IF YOU KNOW WHAT YOU'RE DOING

    grad_t: base update period of the gradient DACs *per channel*;
    e.g. if you're specifying 4 data channels, note that each
    individual channel will be updated with a period of 4*grad_t

    grad_channels: specify how many channels you will be using (3 =
    x,y,z, 4 = x,y,z,z2 etc)

    spi_freq: frequency to run the gradient SPI interface at - must be
    high enough to support your desired grad_t but not so high that
    you experience communication issues. Leave this alone unless you
    know what you're doing.
    """

    def __init__(self,
                 samples=1000,
                 lo_freq=5, # MHz
                 tx_t=0.1, # us, best-effort
                 rx_t=0.5, # us, best-effort
                 instruction_file=None,
                 grad_t=2.5, # us, best-effort
                 grad_channels=4,
                 spi_freq=None,
                 local_grad_board=grad_board): # MHz, best-effort):
        self.samples = samples

        self.lo_freq_bin = int(np.round(lo_freq / fpga_clk_freq_MHz * (1 << 30))) & 0xfffffff0 | 0xf
        self.lo_freq = self.lo_freq_bin * fpga_clk_freq_MHz / (1 << 30)

        self.rx_div = int(np.round(rx_t * fpga_clk_freq_MHz))
        self.rx_t = self.rx_div / fpga_clk_freq_MHz

        # Compensate for factor-of-2 discrepancy in sampling
        self.rx_div_real = self.rx_div // 2 # this is actually what's sent over        
        
        self.tx_div = int(np.round(tx_t * fpga_clk_freq_MHz))
        self.tx_t = self.tx_div / fpga_clk_freq_MHz

        self.instruction_file = instruction_file
        self.asmb = Assembler()
        
        ### Set the gradient controller properties
        grad_clk_t = 0.007 # 7ns period
        self.true_grad_div = np.round(grad_t/grad_clk_t).astype(np.int) # true divider value
        self.grad_div = self.true_grad_div - 4 # what's sent to server
        self.grad_t = grad_clk_t * self.true_grad_div # gradient DAC update period

        spi_cycles_per_tx = 30 # actually 24, but including some overhead
        if spi_freq is not None:
            self.spi_div = np.floor(1 / (spi_freq*grad_clk_t)).astype(np.int) - 1
        else:
            # Auto-decide the SPI freq, to be as low as will work
            assert local_grad_board in ('ocra1', 'gpa-fhdo'), "Unknown gradient board!"
            if local_grad_board == 'ocra1':
                # SPI runs in parallel for each channel
                self.true_spi_div = (self.true_grad_div * grad_channels) // spi_cycles_per_tx
            elif local_grad_board == 'gpa-fhdo':
                # SPI must be written sequentially for each channel
                self.true_spi_div = self.true_grad_div // spi_cycles_per_tx

        self.spi_div = self.true_spi_div - 1

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

    def add_grad(self, vectors):
        """ vectors: list/tuple of real vectors in the range [-1,1] units of full-scale gradient DAC output. Must have the same number of vectors as grad_channels.
        
        Returns the index of the relevant vector, which can be used later when the pulse sequence is being compiled.
        """
                

                 
        assert vec_x.size == vec_y.size == vec_z.size, "Supply equal-length vectors for the three gradients."
        self.grad_offsets.append(self.current_grad_offset)
        self.current_grad_offset += vec_x.size

        try:
            self.grad_data_x = np.hstack( [self.grad_data_x, vec_x] )
        except AttributeError:
            self.grad_data_x = vec_x

        try:
            self.grad_data_y = np.hstack( [self.grad_data_y, vec_y] )
        except AttributeError:            
            self.grad_data_y = vec_y

        try:            
            self.grad_data_z = np.hstack( [self.grad_data_z, vec_z] )
        except AttributeError:            
            self.grad_data_z = vec_z

        return len(self.grad_offsets) - 1

    def compile_tx_data(self):
        """ go through the TX data and prepare binary array to send to the server """
        self.tx_bytes = bytearray(self.tx_data.size * 4)
        if np.any(np.abs(self.tx_data) > 1.0):
            warnings.warn("TX data too large! Overflow will occur.")
        
        tx_i = np.round(32767 * self.tx_data.real).astype(np.uint16)
        tx_q = np.round(32767 * self.tx_data.imag).astype(np.uint16)

        # TODO: find a better way to encode the interleaved bytearray
        self.tx_bytes[::4] = (tx_i & 0xff).astype(np.uint8).tobytes()
        self.tx_bytes[1::4] = (tx_i >> 8).astype(np.uint8).tobytes()
        self.tx_bytes[2::4] = (tx_q & 0xff).astype(np.uint8).tobytes()
        self.tx_bytes[3::4] = (tx_q >> 8).astype(np.uint8).tobytes()

    def compile_grad_data(self):
        """ go through the grad data and prepare binary array to send to the server """
        if not hasattr(self, 'grad_data_x'):
            self.grad_data_x, self.grad_data_y, self.grad_data_z, self.grad_data_z2 = np.array([0]), np.array([0]), np.array([0]), np.array([0])
        self.grad_x_bytes = bytearray(self.grad_data_x.size * 4)
        self.grad_y_bytes = bytearray(self.grad_data_y.size * 4)
        self.grad_z_bytes = bytearray(self.grad_data_z.size * 4)
        for gd in [self.grad_data_x, self.grad_data_y, self.grad_data_z]:
            if np.any(np.abs(gd) > 1.0):
                warnings.warn("Grad data too large! Overflow will occur.")
        
        grx = np.round(32767 * self.grad_data_x).astype(np.uint16)
        gry = np.round(32767 * self.grad_data_y).astype(np.uint16)
        grz = np.round(32767 * self.grad_data_z).astype(np.uint16)        
        
        # TODO: check that this makes sense relative to test_acquire,
        # and find a better way to encode the interleaved bytearray
        self.grad_x_bytes[::4] = ((grx & 0xf) << 4).astype(np.uint8).tobytes()
        self.grad_x_bytes[1::4] = ((grx & 0xff0) >> 4).astype(np.uint8).tobytes()
        self.grad_x_bytes[2::4] = ((grx >> 12) | 0x10).astype(np.uint8).tobytes()
        self.grad_x_bytes[3::4] = np.zeros(self.grad_data_x.size, dtype=np.uint8).tobytes() # wasted?

        self.grad_y_bytes[::4] = ((gry & 0xf) << 4).astype(np.uint8).tobytes()
        self.grad_y_bytes[1::4] = ((gry & 0xff0) >> 4).astype(np.uint8).tobytes()
        self.grad_y_bytes[2::4] = ((gry >> 12) | 0x10).astype(np.uint8).tobytes()
        self.grad_y_bytes[3::4] = np.zeros(self.grad_data_y.size, dtype=np.uint8).tobytes() # wasted?
        
        self.grad_z_bytes[::4] = ((grz & 0xf) << 4).astype(np.uint8).tobytes()
        self.grad_z_bytes[1::4] = ((grz & 0xff0) >> 4).astype(np.uint8).tobytes()
        self.grad_z_bytes[2::4] = ((grz >> 12) | 0x10).astype(np.uint8).tobytes()
        self.grad_z_bytes[3::4] = np.zeros(self.grad_data_z.size, dtype=np.uint8).tobytes() # wasted?

    def compile_instructions(self):
        # For now quite simple (using the ocra assembler)
        # Will use a more advanced approach in the future to avoid having to hand-code the instruction files
        if not hasattr(self, 'instructions'):
            self.instructions = self.asmb.assemble(self.instruction_file)

    def define_instructions(self, instructions):
        self.instructions = instructions

    def compile(self):
        self.compile_tx_data()
        self.compile_grad_data()
        self.compile_instructions()

    def run(self):
        """ compile the TX and grad data, send everything over.
        Returns the resultant data """
        self.compile()
        packet = sc.construct_packet({
            'lo_freq': self.lo_freq_bin,
            'rx_rate': self.rx_div_real,
            'tx_div': self.tx_div,
            'tx_size': self.tx_data.size * 4,
            'raw_tx_data': self.tx_bytes,
            'grad_mem_x': self.grad_x_bytes,
            'grad_mem_y': self.grad_y_bytes,
            'grad_mem_z': self.grad_z_bytes,            
            'seq_data': self.instructions,
            'acq': self.samples})

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect( (ip_address, port) )
        
        reply = sc.send_packet(packet, s)

        try:
            errors = reply[5]['errors']
            for err in errors:
                warnings.warn(err)
        except KeyError:
            pass

        try:
            warns = reply[5]['warnings']
            for war in warns:
                warnings.warn(war)
        except KeyError:
            pass
        
        return np.frombuffer(reply[4]['acq'], np.complex64)

def test_Experiment():
    exp = Experiment(samples=500)
    
    # first TX segment
    t = np.linspace(0, 100, 1001) # goes to 100us, samples every 100ns
    freq = 0.2 # MHz
    tx_x = np.cos(2*np.pi*freq*t) + 1j*np.sin(2*np.pi*freq*t)
    tx_idx = exp.add_tx(tx_x)

    # first gradient segment
    tg = np.linspace(0, 500, 51) # goes to 500us, samples every 10us (sampling rate is fixed right now)
    tmean = 250
    tstd = 100

    grad = np.exp(-(tg-tmean)**2/tstd**2) # Gaussian 
    grad_idx = exp.add_grad(grad, np.zeros_like(grad), np.zeros_like(grad))

    data = exp.run()

    plt.plot(tg, data)
    plt.show()

def test_grad_echo():
    exp = Experiment(samples=1900 + 210, lo_freq=0.5) # sampling rate is off by 2x?
    
    # RF pulse
    t = np.linspace(0, 200, 2001) # goes to 200us, samples every 100ns; length of pulse must be adjusted in grad_echo.txt

    # Square pulse
    if False:
        freq = 0.1 # MHz, offset from LO freq (DC up to a few MHz possible)
        tx_x = np.cos(2*np.pi*freq*t) + 1j*np.sin(2*np.pi*freq*t) # I,Q samples
        tx_idx = exp.add_tx(tx_x) # add the data to the ocra TX memory
    else:
        # sinc pulse
        tx_x = np.sinc( (t - 100) / 25 )
        tx_idx = exp.add_tx(tx_x) # add the data to the ocra TX memory

    # 2nd RF pulse, for testing
    tx_x2 = tx_x*0.5
    tx_idx2 = exp.add_tx(tx_x2)

    # gradient echo; 190 samples total: 50 for first ramp, 140 for second ramp
    grad = np.hstack([
        np.linspace(0, 0.9, 10), np.ones(30), np.linspace(0.9, 0, 10), # first ramp up/down
        np.linspace(0,-0.285, 20), -0.3 * np.ones(100), np.linspace(-0.285, 0, 20)
        ])

    # Correct for DC offset and scaling
    scale = 0.9
    offset = 0.0
    grad_corr = grad*scale + offset
    
    grad_idx = exp.add_grad(grad_corr, grad_corr, grad_corr)
    if False: # set to true if you want to plot the x gradient waveform
        plt.plot(grad_corr);plt.show()

    data = exp.run()

    plt.plot(np.real(data))
    plt.plot(np.imag(data))    
    plt.show()

def test_rx_tx():
    tx_t = 1 # us
    rx_t = 0.5 # us
    tx_pulse1_t = 50 # us
    tx_pulse2_t = 50 # us
    rx_pulse1_t = 50 # us
    tx_offset_f = 0.05 # MHz (50 kHz)
    samples = 300
    # samples=int(rx_pulse1_t // rx_t)
    
    exp = Experiment(samples=samples,
                     lo_freq=2, tx_t=tx_t, rx_t=rx_t,
                     instruction_file="ocra_lib/rx_tx_test.txt")

    t = np.arange(0, tx_pulse1_t, tx_t)
    v_ramp = np.linspace(0.3, 1, t.size)
    tx_y = v_ramp * (
        np.cos(2 * np.pi * tx_offset_f * t)
        + 1j * np.sin(2 * np.pi * tx_offset_f * t) )
    exp.add_tx(tx_y * 0.1) # extra scaling

    if False:
        plt.plot(t, tx_y.real)
        plt.plot(t, tx_y.imag)
        plt.xlabel('time (us)')
        plt.show()

    data = exp.run()

    data_t = np.linspace(0, samples*rx_t, samples)
    plt.plot(data_t, data.real)
    plt.plot(data_t, data.imag)
    plt.xlabel('time (us)')

    plt.savefig('/tmp/images/tx_t_{:.1f}_rx_t_{:.1f}_pr0_{:.1f}_pr1_{:.1f}_samples{:.1f}.png'.format(
        tx_t, tx_t, 1, 50, samples))
    
    plt.show()
        
if __name__ == "__main__":
    # test_Experiment()
    # test_grad_echo()
    test_rx_tx()
