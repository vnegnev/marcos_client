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
                 spi_freq=None, # MHz, best-effort
                 local_grad_board=grad_board,
                 print_infos=True, # show server info messages
                 assert_errors=True, # halt on errors
                 ):
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
        self.true_grad_div = int(np.round(grad_t/grad_clk_t)) # true divider value
        self.grad_div = self.true_grad_div - 4 # what's sent to server
        self.grad_t = grad_clk_t * self.true_grad_div # gradient DAC update period
        self.grad_channels = grad_channels
        assert 0 < grad_channels < 5, "Strange number of grad channels"

        self.grad_board = local_grad_board
        spi_cycles_per_tx = 30 # actually 24, but including some overhead
        if spi_freq is not None:
            self.spi_div = int(np.floor(1 / (spi_freq*grad_clk_t))) - 1
        else:
            # Auto-decide the SPI freq, to be as low as will work
            assert self.grad_board in ('ocra1', 'gpa-fhdo'), "Unknown gradient board!"
            if self.grad_board == 'ocra1':
                # SPI runs in parallel for each channel
                self.true_spi_div = (self.true_grad_div * grad_channels) // spi_cycles_per_tx
            elif self.grad_board == 'gpa-fhdo':
                # SPI must be written sequentially for each channel
                self.true_spi_div = self.true_grad_div // spi_cycles_per_tx

            if self.true_spi_div > 63:
                self.true_spi_div = 63 # slowest SPI clock possible

        self.spi_div = self.true_spi_div - 1
        #self.spi_div = 6;
        print('spi_div ',self.spi_div);
        self.grad_ser = 0x2 if self.grad_board == 'gpa-fhdo' else 0x1 # select which board serialiser is activated on the firmware

        self.print_infos = print_infos
        self.assert_errors = assert_errors

        self.clear_tx()
        self.clear_grad()

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect( (ip_address, port) )

        self.init_gpa()

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

        return reply

    def init_gpa(self):
        """ Setup commands to configure the GPA; only needs to be done once per GPA power-up """
        if self.grad_board == 'ocra1':
            gs = 1
            init_words = [0x00200002, 0x02200002, 0x04200002, 0x07200002]
        else:
            gs = 2
            init_words = [0x00030100, # DAC sync reg
                          0x40850000, 0x400b6000, 0x400d6000, 0x400f6000, 0x40116000] # ADC reset, input ranges for each channel

        # configure grad ctrl divisors
        self.server_command({'grad_div': (self.grad_div, self.spi_div), 'grad_ser': self.grad_ser})
            
        for iw in init_words:
            # direct commands to grad board
            self.server_command({'grad_dir': iw})
    
    def read_gpa_adc(self, channel):
        sc.send_packet(sc.construct_packet({'grad_dir': 0x40c00000 | (channel<<18)}), self.s)
        return sc.send_packet(sc.construct_packet({'grad_adc': 1}), self.s)[4]['grad_adc']
    
    def write_gpa_dac(self, channel, value):
        sc.send_packet(sc.construct_packet({'grad_dir': 0x00080000 | (channel<<16) | int(value)}), self.s) # DAC output
        
    def expected_adc_code(self, dac_code):
        dac_voltage = dac_code/0xFFFF*5
		v_ref = 2.5
		gpa_current_per_volt = 3.75
        gpa_current = (dac_voltage-v_ref) * gpa_current_per_volt
		r_shunt = 0.2
        adc_voltage = gpa_current*r_shunt+v_ref
		adc_gain = 4.096*1.25
        adc_code = int(np.round(adc_voltage*0xFFFF/adc_gain))
        print('DAC code {:d}, DAC voltage {:f}, GPA current {:f}, ADC voltage {:f}, ADC code {:d}'.format(dac_code,dac_voltage,gpa_current,adc_voltage,adc_code))
        return adc_code
        
    def calibrate_gpa_fhdo(self):
        averages = 1
        channel = 0
        dac_values = np.array([0x7000, 0x8000, 0x9000])
        if False:
            np.random.shuffle(dac_values) # to ensure randomised acquisition
        adc_values = np.zeros([dac_values.size, averages])
        for k, dv in enumerate(dac_values):
            self.write_gpa_dac(channel,dv);
            self.expected_adc_code(dv)

            self.read_gpa_adc(channel); # dummy read
            for m in range(averages): 
                adc_values[k, m] = self.read_gpa_adc(channel);
            print('received ADC code {:d}'.format(int((adc_values.sum(1)/averages)[k])))

        self.write_gpa_dac(0,0x8000); # set gradient current back to 0
        self.gpaCalRatios = dac_values/(adc_values.sum(1)/averages);
        plt.plot(dac_values, adc_values.min(1), 'y.')
        plt.plot(dac_values, adc_values.max(1), 'y.')
        plt.plot(dac_values, adc_values.sum(1)/averages, 'b.')
        plt.xlabel('DAC word'); plt.ylabel('ADC word, {:d} averages'.format(averages))
        plt.grid(True)
        plt.show()

    def clear_tx(self):
        self.tx_offsets = []
        self.current_tx_offset = 0
        self.tx_data = np.empty(0)
        self.tx_data_dirty = True        

    def add_tx(self, vec):
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
        self.grad_data = [np.empty(1) for k in range(self.grad_channels)]
        self.grad_offsets = []
        self.current_grad_offset = 0
        self.grad_data_dirty = True
    
    def add_grad(self, vectors):
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
            self.grad_data[k] = np.hstack( [self.grad_data[k], v] )

        return len(self.grad_offsets) - 1

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

        grad_bram_data = np.zeros(self.grad_data[0].size * self.grad_channels, dtype=np.uint32)
        # bytearray(self.grad_data[0].size * self.grad_channels * 4)
        for ch, gd in enumerate(self.grad_data):
            if np.any(np.abs(gd)) > 1.0:
                warnings.warn("Grad data in Ch {:d} outside [-1,1]!".format(ch))

            if self.grad_board == 'ocra1':
                # 2's complement
                # ocra1 16b or 18b -- TODO TEST 18b
                # gr_dacbits = np.round(32767 * gd).astype(np.uint32) & 0xffff
                gr_dacbits = np.round(131071 * gd).astype(np.uint32) & 0x3ffff 
                gr = (gr_dacbits << 2) | 0x00100000
            elif self.grad_board == 'gpa-fhdo':
                # Not 2's complement - 0x0 word is -5V, 0xffff is +5V
                gr_dacbits = np.round(65535 * (gd + 1)).astype(np.uint32) & 0xffff
                gr = gr_dacbits | 0x80000 | (ch << 16) # also handled in gpa_fhdo serialiser, but setting the channel here just in case

            # always broadcast for the final channel
            broadcast = ch == self.grad_channels - 1                

            grad_bram_data[ch::self.grad_channels] = gr | (ch << 25) | (broadcast << 24) # interleave data

            ## initialisation words at address 0 (TODO: test them!)
            if True:
                if self.grad_board == 'ocra1':
                    grad_bram_data[ch] = 0x00200002 | (ch << 25) | (broadcast << 24)

        self.grad_bytes = grad_bram_data.tobytes()
        self.grad_data_dirty = False
        self.grad_data_unsent = True

    def compile_instructions(self):
        # For now quite simple (using the ocra assembler)
        # Will use a more advanced approach in the future to avoid having to hand-code the instruction files
        if not hasattr(self, 'instructions'):
            self.instructions = self.asmb.assemble(self.instruction_file)

    def define_instructions(self, instructions):
        self.instructions = instructions

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
        reply = self.server_command({
            'lo_freq': self.lo_freq_bin,
            'rx_div': self.rx_div_real,
            'tx_div': self.tx_div,
            'tx_size': self.tx_data.size * 4,
            'raw_tx_data': self.tx_bytes,
            'grad_div': (self.grad_div, self.spi_div),
            'grad_ser': self.grad_ser,
            'grad_mem': self.grad_bytes,
            'seq_data': self.instructions,
            'acq': self.samples})
        
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
    # test_Experiment()
    test_grad_echo()
    # test_rx_tx()
