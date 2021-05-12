#!/usr/bin/env python3
#
# Basic toolbox for server operations; wraps up a lot of stuff to avoid the need for hardcoding on the user's side.

import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
import grad_board as gb
import server_comms as sc
import flocompile as fc

import pdb
st = pdb.set_trace

######## TODO: configure the final buffers as well, whether in flocompile or elsewhere

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

    grad_max_update_rate: used to calculate the frequency to run the
    gradient SPI interface at - must be high enough to support your
    maximum gradient sample rate but not so high that you experience
    communication issues. Leave this alone unless you know what you're
    doing.

    print_infos: print debugging messages from server to stdout

    assert_errors: errors returned from the server will be treated as
    exceptions by the class, halting the program

    init_gpa: initialise the GPA during the construction of this class

    """

    def __init__(self,
                 lo_freq=1, # MHz
                 rx_t=1, # us, best-effort
                 seq_dict=None,
                 seq_csv=None,
                 rx_lo=0, # which of LOs (0, 1, 2) to use for each channel
                 grad_max_update_rate=0.2, # MSPS, across all channels in parallel, best-effort
                 gpa_fhdo_offset_time=0, # when GPA-FHDO is used, offset the Y, Z and Z2 gradient times by 1x, 2x and 3x this value to emulate 'simultaneous' updates
                 print_infos=True, # show server info messages
                 assert_errors=True, # halt on errors
                 init_gpa=False, # initialise the GPA (will reset its outputs when the Experiment object is created)
                 initial_wait=None, # initial pause before experiment begins - required to configure the LOs and RX rate; must be at least a few us. Is suitably set based on grad_max_update_rate by default.
                 prev_socket=None, # previously-opened socket, if want to maintain status etc
                 fix_cic_scale=True, # scale the RX data precisely based on the rate being used; otherwise a 2x variation possible in data amplitude based on rate
                 set_cic_shift=False, # program the CIC internal bit shift to maintain the gain within a factor of 2 independent of rate; required if the open-source CIC is used in the design
                 halt_and_reset=False, # upon connecting to the server, halt any existing sequences that may be running
                 flush_old_rx=False, # when debugging or developing new code, you may accidentally fill up the RX FIFOs - they will not automatically be cleared in case there is important data inside. Setting this true will always read them out and clear them before running a sequence. More advanced manual code can read RX from existing sequences.
                 ):

        # create socket early so that destructor works
        if prev_socket is None:
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._s.connect( (ip_address, port) )
        else:
            self._s = prev_socket

        # extend lo_freq to 3 elements
        if not hasattr(lo_freq, "__len__"):
            lo_freq = lo_freq, lo_freq, lo_freq # extend to 3 elements
        elif len(lo_freq) < 3:
            lo_freq = lo_freq[0], lo_freq[1], lo_freq[0] # extend from 2 to 3 elements

        self._dds_phase_steps = np.round(2**31 / fpga_clk_freq_MHz * np.array(lo_freq)).astype(np.uint32)
        self._lo_freqs = self._dds_phase_steps * fpga_clk_freq_MHz / (2 ** 31) # real LO freqs

        if not hasattr(rx_t, "__len__"):
            rx_t = rx_t, rx_t # extend to 2 elements

        # TODO: enable variable rates during a single TR
        self._rx_divs = np.round(np.array(rx_t) * fpga_clk_freq_MHz).astype(np.uint32)
        self._rx_ts = self._rx_divs / fpga_clk_freq_MHz

        if not hasattr(rx_lo, "__len__"):
            rx_lo = rx_lo, rx_lo # extend to 2 elements
        self._rx_lo = rx_lo

        assert grad_board in ('ocra1', 'gpa-fhdo'), "Unknown gradient board!"
        if grad_board == 'ocra1':
            gradb_class = gb.OCRA1
            self._gpa_fhdo_offset_time = 0
        else:
            gradb_class = gb.GPAFHDO
            self._gpa_fhdo_offset_time = gpa_fhdo_offset_time
        self.gradb = gradb_class(self.server_command, grad_max_update_rate)

        if initial_wait is None:
            # auto-set the initial wait to be long enough for initial gradient configuration to finish, plus 1us for miscellaneous startup
            self._initial_wait = 1 + 1/grad_max_update_rate

        assert (seq_csv is None) or (seq_dict is None), "Cannot supply both a sequence dictionary and a CSV file."
        self._csv = None
        self._seq = None
        if seq_dict is not None:
            self.add_flodict(seq_dict)
        elif seq_csv is not None:
            self._csv = seq_csv # None unless a CSV was supplied

        self._print_infos = print_infos
        self._assert_errors = assert_errors

        if init_gpa:
            self.gradb.init_hw()

        if halt_and_reset:
            halted = sc.command({'halt_and_reset': 0}, self._s)[0][4]['halt_and_reset']
            assert halted, "Could not halt the execution of an existing sequence. Please file a bug report."

        self._fix_cic_scale = fix_cic_scale
        self._set_cic_shift = set_cic_shift
        self._flush_old_rx = flush_old_rx

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

    def flo2int(self, seq_dict):
        """Convert a floating-point sequence dictionary to an integer binary
        dictionary"""

        intdict = {}

        ## Various functions to handle the conversion
        def times_us(farr):
            """ farr: float array, times in us units; [0, inf) """
            return np.round(fpga_clk_freq_MHz * farr).astype(np.int64) # negative values will get rejected at a later stage

        def tx_real(farr):
            """ farr: float array, [-1, 1] """
            return np.round(32767 * farr).astype(np.uint16)

        def tx_complex(farr):
            """ farr: complex float array, [-1-1j, 1+1j] -- returns a tuple """
            idata, qdata = farr.real, farr.imag
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
            elif key in ['grad_vx', 'grad_vy', 'grad_vz', 'grad_vz2',
                         'fhdo_vx', 'fhdo_vy', 'fhdo_vz', 'fhdo_vz2',
                         'ocra1_vx', 'ocra1_vy', 'ocra1_vz', 'ocra1_vz2']:
                # flocompile will figure out whether the key matches the selected grad board
                keyb, channel = self.gradb.key_convert(key)
                keybin = keyb, # tuple
                valbin = self.gradb.float2bin(vals, channel),

                # hack to de-synchronise GPA-FHDO outputs, to give the
                # user the illusion of being able to output on several
                # channels in parallel
                if self._gpa_fhdo_offset_time:
                    tbin = times_us(times + channel*self._gpa_fhdo_offset_time + self._initial_wait)

            elif key in ['rx0_rate', 'rx1_rate']:
                keybin = key,
                valbin = vals.astype(np.uint16),
            elif key in ['rx0_rate_valid', 'rx1_rate_valid', 'rx0_rst_n', 'rx1_rst_n', 'rx0_en', 'rx1_en', 'tx_gate', 'rx_gate', 'trig_out']:
                keybin = key,
                # binary-valued data
                valbin = vals.astype(np.int32),
                for vb in valbin:
                    assert np.all( (0 <= vb) & (vb <= 1) ), "Binary columns must be [0,1] or [False, True] valued"
            elif key in ['leds']:
                keybin = key,
                valbin = vals.astype(np.int32),
            else:
                warnings.warn("Unknown flocra experiment dictionary key: " + key)
                continue

            for k, v in zip(keybin, valbin):
                intdict[k] = (tbin, v)

        return intdict

    def add_intdict(self, seq_intdict, append=True):
        """ Add an integer-format dictionary to the sequence, or replace the old one """
        if self._seq is None:
            self._seq = {}

        for name, sb in seq_intdict.items():
            if name in self._seq.keys() and append:
                a, b = self._seq[name]
                self._seq[name] = ( np.append(a, sb[0]), np.append(b, sb[1]) )
            else:
                self._seq[name] = sb

    def add_flodict(self, flodict, append=True):
        """ Add a floating-point dictionary to the sequence """
        assert self._csv is None, "Cannot replace the dictionary for an Experiment class created from a CSV"
        self.add_intdict(self.flo2int(flodict), append)
        self._seq_compiled = False

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
        initial_cfg = {# 'rx0_rst_n': ( np.array([tstart, tstart + 2*rx_wait]), np.array([1, 0]) ),
                       # 'rx1_rst_n': ( np.array([tstart, tstart + 2*rx_wait]), np.array([1, 0]) ),
                       'rx0_rst_n': ( np.array([tstart]), np.array([1]) ),
                       'rx1_rst_n': ( np.array([tstart]), np.array([1]) ),
                       'lo0_freq': ( np.array([tstart]), np.array([self._dds_phase_steps[0]]) ),
                       'lo1_freq': ( np.array([tstart]), np.array([self._dds_phase_steps[1]]) ),
                       'lo2_freq': ( np.array([tstart]), np.array([self._dds_phase_steps[2]]) ),
                       'lo0_rst': ( np.array([tstart, tstart + 1]), np.array([1, 0]) ),
                       'lo1_rst': ( np.array([tstart, tstart + 1]), np.array([1, 0]) ),
                       'lo2_rst': ( np.array([tstart, tstart + 1]), np.array([1, 0]) )
                       }

        # Set CIC decimation rate and internal shift, if necessary, and calculate CIC scale correction
        rx0_words, self._rx0_cic_factor = fc.cic_words(self._rx_divs[0], self._set_cic_shift)
        rx1_words, self._rx1_cic_factor = fc.cic_words(self._rx_divs[1], self._set_cic_shift)
        if not self._fix_cic_scale: # clear the correction factor
            self._rx0_cic_factor = 1
            self._rx1_cic_factor = 1
        rx0r_st = tstart + rx_wait
        wds = len(rx0_words)
        ar0 = np.arange(wds, dtype=int)
        ar1 = np.arange(wds + 1, dtype=int)
        ow = np.ones(wds + 1, dtype=int)
        ow[-1] = 0
        initial_cfg.update({
            'rx0_rate': ( tstart + rx_wait + ar0, np.array(rx0_words) ),
            'rx1_rate': ( tstart + rx_wait + ar0, np.array(rx1_words) ),
            'rx0_rate_valid': ( tstart + rx_wait + ar1, ow),
            'rx1_rate_valid': ( tstart + rx_wait + ar1, ow),
        })

        # LO source configuration (only set non-default values if necessary)
        if self._rx_lo[0] != 0:
            initial_cfg.update({ 'rx0_lo': ( np.array([tstart]), np.array([self._rx_lo[0]]) ) })
        if self._rx_lo[1] != 0:
            initial_cfg.update({ 'rx1_lo': ( np.array([tstart]), np.array([self._rx_lo[1]]) ) })

        self.add_intdict(initial_cfg)

        self._machine_code = np.array( fc.dict2bin(self._seq,
                                             self.gradb.bin_config['initial_bufs'],
                                             self.gradb.bin_config['latencies'], # TODO: can add extra manipulation here, e.g. add to another array etc
                                             ), dtype=np.uint32 )

    def run(self):
        """ compile the TX and grad data, send everything over.
        Returns the resultant data """

        if self._seq_compiled is False:
            self.compile()

        if self._flush_old_rx:
            rx_data_old, _ = sc.command({'read_rx': 0}, self._s)
            # TODO: do something with RX data previously collected by the server

        rx_data, msgs = sc.command({'run_seq': self._machine_code.tobytes()}, self._s)

        rxd = rx_data[4]['run_seq']
        rxd_iq = {}

        # (1 << 24) just for the int->float conversion to be reasonable - exact value doesn't matter for now
        rx0_norm_factor = self._rx0_cic_factor / (1 << 24)
        rx1_norm_factor = self._rx0_cic_factor / (1 << 24)

        try:
            rxd_iq['rx0'] = rx0_norm_factor * ( np.array(rxd['rx0_i']).astype(np.int32).astype(float) + \
                             1j * np.array(rxd['rx0_q']).astype(np.int32).astype(float) )
        except (KeyError, TypeError):
            pass

        try:
            rxd_iq['rx1'] = rx1_norm_factor * ( np.array(rxd['rx1_i']).astype(np.int32).astype(float) + \
                             1j * np.array(rxd['rx1_q']).astype(np.int32).astype(float) )
        except (KeyError, TypeError):
            pass

        return rxd_iq, msgs

    def close_server(self, only_if_sim=False):
        ## Either always close server, or only close server if it's a simulation
        if not only_if_sim or sc.command({'are_you_real':0}, self._s)[0][4]['are_you_real'] == "simulation":
            sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), self._s)

def test_rx_scaling(lo_freq=0.5, rf_amp=0.5, rf_steps=True, rx_time=50, rx_periods=[600], rx_padding=20, plot_rx=False):

    expt = Experiment(lo_freq=lo_freq, rx_t=rx_periods[0] / fpga_clk_freq_MHz, fix_cic_scale=False, set_cic_shift=False, flush_old_rx=True)
    tr_t = 0
    tr_period = rx_time + rx_padding
    rx_lengths = []

    def single_pulse_tr(tstart, rx_period=20):
        if rf_steps:
            tx_shape = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0]) * rf_amp
            tx_times = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) * (rx_time - 2*rx_padding)
        else:
            tx_shape = np.array([1, 0]) * rf_amp
            tx_times = np.array([0, 1]) * (rx_time - 2*rx_padding)
        tx_seq = ( tstart + 5 + rx_padding + tx_times, tx_shape )
        rx_seq = ( tstart + 5 + np.array([0, rx_time]), np.array([1, 0]) )

        # Rate adjustment
        set_cic_shift = expt._set_cic_shift
        rx_words, _ = fc.cic_words(rx_period, set_cic_shift)
        rx_wait = 100
        rxr_st = tstart + rx_wait
        wds = len(rx_words)
        ar0 = np.arange(wds, dtype=int)
        ar1 = np.arange(wds + 1, dtype=int)
        ow = np.ones(wds + 1, dtype=int)
        ow[-1] = 0
        rate_seq = ( tstart + (rx_wait + ar0) / fpga_clk_freq_MHz, np.array(rx_words) )
        rate_en_seq = ( tstart + (rx_wait + ar1) / fpga_clk_freq_MHz, ow )

        value_dict = {
            'tx0': tx_seq, 'tx1': tx_seq,
            # 'rx0_en': rx_seq,
            'rx1_en': rx_seq,
            'rx0_rate': rate_seq, 'rx1_rate': rate_seq,
            'rx0_rate_valid': rate_en_seq, 'rx1_rate_valid': rate_en_seq,
            }

        # if rx_period % 2:
        #     value_dict.pop('rx0_rate_valid', None)
        #     value_dict.pop('rx1_rate_valid', None)

        return value_dict

    for rt in rx_periods:
        expt.add_flodict( single_pulse_tr( tr_t , rt) )
        tr_t += tr_period
        rx_lengths.append( int(rx_time * fpga_clk_freq_MHz / rt) + 1)

    rxd, msgs = expt.run()
    expt.close_server(True)

    # rx_lengths_a = np.array(rx_lengths)
    # print("RX lengths from calculation: ", rx_lengths)

    # rx0_mag = np.abs(rxd['rx0'])
    rx1_mag = np.abs(rxd['rx1'])
    plt.plot(rx1_mag);plt.show()

    pulse_means = []

    if plot_rx:
        initial_excess = 5 # amount to throw away from the start of each acquisition
        M = 0

        plt.figure(figsize=(12, 6))
        plt.xlabel('RX sample')
        plt.ylabel('RF magnitude')
        for k, l in zip(rx_periods, rx_lengths):
            x = np.arange(M + initial_excess, M+l)
            y = np.array(rx1_mag[M + initial_excess : M+l])

            _, cic_correction = fc.cic_words(k)
            ysc = y * cic_correction

            # avoid off-by-1 errors - bit of a hack
            if len(ysc) != len(x):
                x = x[:-1]

            if True:
                plt.plot(x, ysc)
            M += l
            # pulse_means.append( y[y > 0.5*y.max()].mean() )
            # pulse_means.append( y.mean() )
            pulse_means.append( ysc.max() )

        plt.figure(figsize=(8,7))
        pma = np.array(pulse_means)
        plt.subplot(2,1,1)
        plt.plot(rx_periods, pma/pma.mean())
        plt.xlabel('RX decimation')
        plt.ylabel('RX peak amp')
        plt.grid(True)
        # plt.subplot(3,1,2)
        # plt.plot(rx_periods, rx_lengths)
        # plt.legend(['rx samples'])
        # plt.xlabel('RX decimation')
        # plt.grid(True)
        plt.subplot(2,1,2)
        _, cic_corrections = fc.cic_words(rx_periods)
        plt.plot(rx_periods, cic_corrections )
        plt.xlabel('RX decimation')
        plt.ylabel('RX CIC correction factor')
        plt.grid(True)
        plt.show()

def test_gpa_calibration():
    expt = Experiment(init_gpa=True)
    expt.gradb.calibrate(channels=[0], max_current=0.7, num_calibration_points=30, averages=5, settle_time=0.005, poly_degree=5)
    expt.gradb.calibrate(channels=[0], max_current=0.7, num_calibration_points=30, averages=1, test_cal=True)

if __name__ == "__main__":
    if False:
        test_rx_scaling(lo_freq=8,
                        rf_amp=0.5,
                        rf_steps=False,
                        rx_time=60,
                        rx_padding=20,
                        # rx_periods = np.ones(10, dtype=int)*30,
                        # rx_periods=np.arange(4, 400, 1),
                        rx_periods=np.arange(4, 400, 1),
                        # rx_periods=np.arange(10, 400, 1),
                        plot_rx=True)

    if True:
        test_gpa_calibration()
