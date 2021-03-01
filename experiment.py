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
                 seq_dict=None,
                 seq_csv=None,
                 rx_lo=0, # which of LOs (0, 1, 2) to use for each channel
                 spi_freq=5, # MHz, best-effort -- default supports at least 100 ksps
                 local_grad_board="auto", # auto uses the local_config.py value, otherwise can be overridden here
                 print_infos=True, # show server info messages
                 assert_errors=True, # halt on errors
                 init_gpa=False, # initialise the GPA (will reset its outputs when the Experiment object is created)
                 initial_wait=1, # initial pause before experiment begins - required to configure the LOs and RX rate; must be at least 1us
                 prev_socket=None # previously-opened socket, if want to maintain status etc
                 ):

        # create socket early so that destructor works
        if prev_socket is None:
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self._s = prev_socket
        
        # extend lo_freq to 3 elements
        if type(lo_freq) in (int, float):
            lo_freq = lo_freq, lo_freq, lo_freq # extend to 3 elements
        elif len(lo_freq) < 3:
            lo_freq = lo_freq[0], lo_freq[1], lo_freq[0] # extend from 2 to 3 elements

        self._dds_phase_steps = np.round(2**31 / fpga_clk_freq_MHz * np.array(lo_freq)).astype(np.uint32)
        self._lo_freqs = self._dds_phase_steps * fpga_clk_freq_MHz / (2 ** 31) # real LO freqs

        if type(rx_t) in (int, float):
            rx_t = rx_t, rx_t # extend to 2 elements

        # TODO: enable variable rates during a single TR
        self._rx_divs = np.round(np.array(rx_t) * fpga_clk_freq_MHz).astype(np.uint32)
        self._rx_ts = self._rx_divs / fpga_clk_freq_MHz

        if type(rx_lo) == int: 
            rx_lo = rx_lo, rx_lo # extend to 2 elements
        self._rx_lo = rx_lo

        self._initial_wait = initial_wait

        assert (seq_csv is None) or (seq_dict is None), "Cannot supply both a sequence dictionary and a CSV file."
        if seq_dict is not None:
            self._csv = None
            self.replace_dict(seq_dict)
        else:
            self._csv = seq_csv # None unless a CSV was supplied
            
        self._s.connect( (ip_address, port) )
        
        if local_grad_board == "auto":
            local_grad_board = grad_board
            
        assert local_grad_board in ('ocra1', 'gpa-fhdo'), "Unknown gradient board!"
        if local_grad_board == 'ocra1':
            gradb_class = gb.OCRA1
        else:
            gradb_class = gb.GPAFHDO
        self.gradb = gradb_class(self.server_command, spi_freq)

        self._print_infos = print_infos
        self._assert_errors = assert_errors

        if init_gpa:
            self.gradb.init_hw()

    def __del__(self):
        self._s.close()

    def server_command(self, server_dict):
        packet = sc.construct_packet(server_dict)
        reply = sc.send_packet(packet, self._s)
        return_status = reply[5]
        
        if self._print_infos and 'infos' in return_status:
            print("Server info:")
            for k in return_status['infos']:
                print(k)

        if 'warnings' in return_status:
            for k in return_status['warnings']:
                warnings.warn(k)

        if 'errors' in return_status:            
            if self._assert_errors:
                assert 'errors' not in return_status, return_status['errors'][0]
            else:
                for k in return_status['errors']:
                    warnings.warn("ERROR: " + k)

        return reply, return_status

    def replace_dict(self, seq_dict):
        assert self._csv is None, "Cannot replace the dictionary for an Experiment class created from a CSV"
        self._seq = {}
        self._seq_compiled = False

        ## Various functions to handle the conversion
        def times_us(farr):
            """ farr: float array, times in us units; [0, inf) """
            return np.round(fpga_clk_freq_MHz * farr).astype(np.int64) # negative values will get rejected at a later stage
        
        def tx_real(farr):
            """ farr: float array, [-1, 1] """
            return np.round(32767 * farr).astype(np.uint16)

        def tx_complex(farr):
            """ farr: complex float array, [-1-1j, 1+1j] -- returns a tuple """
            idata, qdata = farr.real(), farr.imag()
            return tx_real(idata), tx_real(qdata)

        for key, (times, vals) in seq_dict.items():
            # each possible dictionary entry returns a tuple (even if one element) for the binary dictionary to send to flocompile
            tbin = times_us(times + self._initial_wait)
            if key in ['tx0_i', 'tx0_q', 'tx1_i', 'tx1_q']:
                valbin = tx_real(vals),
                keybin = key,
            elif key in ['tx0', 'tx1']:
                valbin = tx_complex(vals)
                keybin = key + '_i', key + '_q'
            elif key in ['grad_vx', 'grad_vy', 'grad_vz', 'grad_vz2', 'fhdo_vx', 'fhdo_vy', 'fhdo_vz', 'fhdo_vz2']:
                # flocompile will figure out whether the key matches the selected grad board
                keybin = key,
                valbin = self.gradb.float2bin(vals),
            elif key in ['rx0_rst_n', 'rx1_rst_n', 'tx_gate', 'rx_gate', 'trig_out']:
                keybin = key,
                # binary-valued data
                valbin = vals.astype(np.int32),
                assert np.all( (0 <= valbin) & (valbin <= 1) ), "Binary columns must be [0,1] or [False, True] valued"
            elif key in ['leds']:
                keybin = key,
                valbin = vals.astype(np.uint8), # 8-bit value

            for k, v in zip(keybin, valbin):
                self._seq[k] = (tbin, v)

    def compile(self):
        """Convert either dictionary or CSV file into machine code, with
        extra machine code at the start to ensure the system is initialised to
        the correct state.

        Initially, configure the RX rates and set the LEDs.
        Remainder of the sequence will be as programmed.
        """

        # RX and LO configuration
        tstart = 50 # cycles before doing anything
        rx_wait = 50 # cycles to run RX before setting rate, then later resetting again
        initial_cfg = {'rx0_rate': ( np.array([tstart + rx_wait]), np.array([self._rx_divs[0]]) ),
                       'rx1_rate': ( np.array([tstart + rx_wait]), np.array([self._rx_divs[1]]) ),
                       'rx0_rate_valid': ( np.array([tstart + rx_wait, tstart + rx_wait + 1]), np.array([1, 0]) ),
                       'rx1_rate_valid': ( np.array([tstart + rx_wait, tstart + rx_wait + 1]), np.array([1, 0]) ),
                       'rx0_rst_n': ( np.array([tstart, tstart + 2*rx_wait]), np.array([1, 0]) ),
                       'rx1_rst_n': ( np.array([tstart, tstart + 2*rx_wait]), np.array([1, 0]) ),
                       'lo0_freq': ( np.array([tstart]), np.array([self._dds_phase_steps[0]]) ),
                       'lo1_freq': ( np.array([tstart]), np.array([self._dds_phase_steps[1]]) ),
                       'lo2_freq': ( np.array([tstart]), np.array([self._dds_phase_steps[2]]) ),
                       'lo0_rst': ( np.array([tstart, tstart + 1]), np.array([1, 0]) ),
                       'lo1_rst': ( np.array([tstart, tstart + 1]), np.array([1, 0]) ),
                       'lo2_rst': ( np.array([tstart, tstart + 1]), np.array([1, 0]) ),
                       'rx0_lo': ( np.array([tstart]), np.array([self._rx_lo[0]]) ),
                       'rx1_lo': ( np.array([tstart]), np.array([self._rx_lo[1]]) ),
                       }
        
        self._seq.update(initial_cfg)
        self._binseq = np.array( fc.dict2bin(self._seq,
                                             self.gradb.bin_config['initial_bufs'],
                                             self.gradb.bin_config['latencies'], # TODO: can add extra manipulation here, e.g. add to another array etc
                                             ), dtype=np.uint32 )
            
    def run(self):
        """ compile the TX and grad data, send everything over.
        Returns the resultant data """

        if self._seq_compiled is False:
            self.compile()
        
        rx_data, msgs = sc.command({'run_seq': self._binseq.tobytes()}, self._s)
        st()
        
        return np.frombuffer(reply[4]['acq'], np.complex64), status

def test_Experiment():
    sd = {'tx0_i': ( np.array([1,2,10]), np.array([-0.5, 0.5, 0.9]) )}
    exp = Experiment(lo_freq=1, seq_dict=sd)
    exp.run()
    st()
    
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
