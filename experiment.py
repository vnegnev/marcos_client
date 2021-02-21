#!/usr/bin/env python3
# 
# Basic toolbox for server operations; wraps up a lot of stuff to avoid the need for hardcoding on the user's side.

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
import grad_board as gb
import server_comms as sc
import flocompile as fc

class Experiment:
    """Wrapper class for managing an entire experimental sequence 

    lo_freq: local oscillator frequencies, MHz: either single float,
    iterable of two or iterable of three values. Control three
    independent LOs. At least a single float must be supplied

    rx_t: RF RX sampling time/s in microseconds; single float or tuple
    of two.

    rx_lo: RX local oscillator sources (integers): 0 - 2 correspond to
    the three LO NCOs, 3 is DC. Single integer or tuple of two.  By
    default will use local oscillator 0 for both RX channels, unless
    otherwise specified.

    csv_file: path to a CSV execution file, which will be
    compiled with flocompile.py . If this is not supplied, the
    sequence bytecode should be supplied manually using
    define_sequence() before run() is called.

    OPTIONAL PARAMETERS - ONLY ALTER IF YOU KNOW WHAT YOU'RE DOING

    spi_freq: frequency to run the gradient SPI interface at - must be
    high enough to support your maximum gradient sample rate but not
    so high that you experience communication issues. Leave this alone
    unless you know what you're doing.

    local_grad_board: override local_config.py setting

    print_infos: print debugging messages from server to stdout

    assert_errors: errors returned from the server will be treated as
    exceptions by the class, halting the program

    init_gpa: initialise the GPA during the construction of this class
    """

    def __init__(self,
                 lo_freq, # MHz
                 rx_t=1, # us, best-effort
                 csv_file=None,
                 rx_lo=0,
                 spi_freq=None, # MHz, best-effort
                 local_grad_board="auto", # auto uses the local_config.py value, otherwise can be overridden here
                 print_infos=True, # show server info messages
                 assert_errors=True, # halt on errors
                 init_gpa=False, # initialise the GPA (will reset its outputs when the Experiment object is created)
                 ):

        # extend lo_freq to 3 elements
        if type(lo_freq) in (int, float):
            lo_freq = lo_freq, lo_freq, lo_freq # extend to 3 elements
        elif len(lo_freq) < 3:
            lo_freq = lo_freq[0], lo_freq[1], lo_freq[0] # extend from 2 to 3 elements

        self.dds_phase_steps = np.round(2**31 / fpga_clk_freq_MHz * np.array(lo_freq)).astype(np.uint32)
        self.lo_freqs = self.dds_phase_steps * fpga_clk_freq_MHz / (2 ** 31) # real LO freqs

        if type(rx_t) in (int, float):
            rx_t = rx_t, rx_t # extend to 2 elements
        
        self.rx_divs = np.round(np.array(rx_t) * fpga_clk_freq_MHz).astype(np.uint32)
        self.rx_ts = self.rx_divs / fpga_clk_freq_MHz

        if type(rx_lo) == int: 
            rx_lo = rx_lo, rx_lo # extend to 2 elements
        
        self.csv_file = csv_file

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect( (ip_address, port) )
        
        if local_grad_board == "auto":
            local_grad_board = grad_board

        ## CONTINUE HERE, FIGURE OUT HOW TO HANDLE LOCAL CALIBRATIONS
        ## MAYBE SEPARATE EVENT ARRAYS?
            
        assert local_grad_board in ('ocra1', 'gpa-fhdo'), "Unknown gradient board!"
        if local_grad_board == 'ocra1':
            gradb_class = gb.OCRA1
        else:
            gradb_class = gb.GPAFHDO
        self.gradb = gradb_class(self.server_command, spi_freq)

        self.print_infos = print_infos
        self.assert_errors = assert_errors

        if init_gpa:
            self.gradb.init_hw()

    def __del__(self):
        self.s.close()

    def server_command(self, server_dict):
        packet = sc.construct_packet(server_dict)
        reply = sc.send_packet(packet, self.s)
        return_status = reply[5]
        
        if self.print_infos and 'infos' in return_status:
            print("Server info:")
            for k in return_status['infos']:
                print(k)

        if 'warnings' in return_status:
            for k in return_status['warnings']:
                warnings.warn(k)

        if 'errors' in return_status:            
            if self.assert_errors:
                assert 'errors' not in return_status, return_status['errors'][0]
            else:
                for k in return_status['errors']:
                    warnings.warn("ERROR: " + k)

        return reply, return_status

    def clear_tx(self):
        self.tx_offsets = []
        self.current_tx_offset = 0
        self.tx_data = np.empty(0)
        self.tx_data_dirty = True        

    def add_tx(self, vec, channel):
        """ vec: complex vector in the I,Q range [-1,1] and [-j,j]; units of full-scale RF DAC output.
        (Note that the magnitude of each element must be <= 1, i.e. np.abs(1+1j) is sqrt(2) and thus too high.)
        
        Returns the index of the relevant vector, which can be used later when the pulse sequence is being compiled.
        """
        self.tx_data_dirty = True
        self.tx_offsets.append(self.current_tx_offset)
        self.current_tx_offset += vec.size
        self.tx_data = np.hstack( [self.tx_data, vec] )

        return len(self.tx_offsets) - 1

    def clear_grad(self):
        self.grad_data = [np.empty(0) for k in range(self.grad_channels)]
        self.grad_offsets = []
        self.current_grad_offset = 0
        self.grad_data_dirty = True
    
    def add_grad(self, vec, channel):
        """ vectors: list/tuple of real vectors in the range [-1,1] units of full-scale gradient DAC output. Must have the same number of vectors as grad_channels.
        
        Returns the index of the relevant vectors, which can be used later when the pulse sequence is being compiled.
        """
        self.grad_data_dirty = True
        assert len(vectors) == self.grad_channels, "One vector required for each gradient channel"
        vsz = vectors[0].size
        for v in vectors[1:]:
            assert v.size == vsz, "Supply equal-length vectors for all the gradients."
                 
        self.grad_offsets.append(self.current_grad_offset)
        self.current_grad_offset += vsz

        if not hasattr(self, 'grad_data'):
            self.clear_grad()

        for k, v in enumerate(vectors):
            assert np.all( (-1 <= v) & (v <= 1) ), "Grad data out of range"
            self.grad_data[k] = np.hstack( [self.grad_data[k], v] )

        return len(self.grad_offsets) - 1

    def compile(self):
        ## TODO: write this -- send the local update dictionary to dict2bin in flocompile, save the resultant instructions

        ## TODO: integrate this
        def tx_f2i(farr):
            """ farr: float array, [-1, 1] """
            return np.round(32767 * farr).astype(np.uint16)        
        

    def compile_tx_data(self):
        """ go through the TX data and prepare binary array to send to the server """

        if self.tx_data.size == 0:
            self.tx_data = np.array([0])
            
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
        self.tx_data_dirty = False

    def compile_grad_data(self):
        """ go through the grad data and prepare binary array to send to the server """
        if not hasattr(self, 'grad_data'):
            self.clear_grad()

        if self.grad_data[0].size == 0:
            self.grad_data = [np.array([0]) for k in range(self.grad_channels)]

        grad_bram_data = self.gradb.float2bin(self.grad_data) # grad board-specific transformation
            
        self.grad_bytes = grad_bram_data.tobytes()
        self.grad_data_dirty = False
        self.grad_data_unsent = True

    def compile_instructions(self):
        # For now quite simple (using the ocra assembler)
        # Will use a more advanced approach in the future to avoid having to hand-code the instruction files
        if not hasattr(self, 'instructions'):
            self.instructions = self.asmb.assemble(self.instruction_file)

    def define_bytecode(self, bytecode):
        self.bytecode = bytecode

    def auto_compile(self):
        self.compile_instructions()
        if self.tx_data_dirty:
            self.compile_tx_data()
        if self.grad_data_dirty:
            self.compile_grad_data()

    def run(self):
        """ compile the TX and grad data, send everything over.
        Returns the resultant data """
        self.auto_compile()        
        reply, status = self.server_command({
            'lo_freq': self.lo_freq_bin,
            'rx_div': self.rx_div_real,
            'tx_div': self.tx_div,
            'tx_size': self.tx_data.size * 4,
            'raw_tx_data': self.tx_bytes,
            'grad_div': (self.gradb.grad_div, self.gradb.spi_div),
            'grad_ser': self.gradb.grad_ser,
            'grad_mem': self.grad_bytes,
            'seq_data': self.instructions,
            'acq_rlim': self.acq_retry_limit,
            'acq': self.samples})
        
        return np.frombuffer(reply[4]['acq'], np.complex64), status

def test_Experiment():
    exp = Experiment(lo_freq=1)
    
    # # first TX segment
    # t = np.linspace(0, 100, 1001) # goes to 100us, samples every 100ns
    # freq = 0.2 # MHz
    # tx_x = np.cos(2*np.pi*freq*t) + 1j*np.sin(2*np.pi*freq*t)
    # tx_idx = exp.add_tx(tx_x)

    # # first gradient segment
    # tg = np.linspace(0, 500, 51) # goes to 500us, samples every 10us (sampling rate is fixed right now)
    # tmean = 250
    # tstd = 100

    # grad = np.exp(-(tg-tmean)**2/tstd**2) # Gaussian 
    # grad_idx = exp.add_grad(grad, np.zeros_like(grad), np.zeros_like(grad))

    # data = exp.run()

    # plt.plot(tg, data)
    # plt.show()

def test_grad_echo():
    exp = Experiment(samples=1900 + 210,
                     lo_freq=0.5,
                     grad_channels=3,
                     instruction_file='ocra_lib/grad_echo.txt',
                     grad_t=1.8) # sampling rate is off by 2x?
    
    # RF pulse
    t = np.linspace(0, 200, 2001) # goes to 200us, samples every 100ns; length of pulse must be adjusted in grad_echo.txt

    if False:
        # square pulse at an offset frequency
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
    
    grad_idx = exp.add_grad([grad_corr, grad_corr, grad_corr])
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
    test_Experiment()
    # test_grad_echo()
    # test_rx_tx()
