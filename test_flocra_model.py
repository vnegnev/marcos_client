#!/usr/bin/env python3
#
# Integrated test of the flocra server, HDL and compiler.
#
# Script compiles and runs the flocra simulation (server + HDL model)
# and sends it various binaries generated using the flocra client
# compiler, then compares the simulated hardware output against the
# expected output.
# 
# To run a single test, use e.g.:
# python -m unittest test_flocra_model.Modeltest.test_many_quick

import sys, os, subprocess, warnings, socket, unittest, time
import numpy as np
import matplotlib.pyplot as plt

import server_comms as sc

import flocompile as fc
import experiment as exp

import pdb
st = pdb.set_trace
ip_address = "localhost"
port = 11111

# simulation configuration
flocra_sim_path = os.path.join("..", "flocra")
flocra_sim_csv = os.path.join("/tmp", "flocra_sim.csv")

# Set to True to debug with GTKWave -- just do one test at a time!
flocra_sim_fst_dump = False
flocra_sim_fst = os.path.join("/tmp", "flocra_sim.fst")

# Arguments for compare_csv when running gradient tests
fhd_config = {
    'initial_bufs': np.array([
        # see flocra.sv, gradient control lines (lines 186-190, 05.02.2021)
        # strobe for both LSB and LSB, reset_n = 1, spi div = 10, grad board select (1 = ocra1, 2 = gpa-fhdo)
        (1 << 9) | (1 << 8) | (10 << 2) | 2,
        0, 0,
        0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0], dtype=np.uint16),
    'latencies': np.array([
        0, 276, 276, # grad latencies match SPI div
        0, 0, # rx
        0, 0, 0, 0, # tx
        0, 0, 0, 0, 0, 0, # lo phase
        0, 0 # gates and LEDs, RX config
    ], dtype=np.uint16)}

oc1_config = {
    'initial_bufs': np.array([
        # see flocra.sv, gradient control lines (lines 186-190, 05.02.2021)
        # strobe for both LSB and LSB, reset_n = 1, spi div = 10, grad board select (1 = ocra1, 2 = gpa-fhdo)
        (1 << 9) | (1 << 8) | (10 << 2) | 1,
        0, 0,
        0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0], dtype=np.uint16),
    'latencies': np.array([
        0, 268, 268, # grad latencies match SPI div
        0, 0, # rx
        0, 0, 0, 0, # tx
        0, 0, 0, 0, 0, 0, # lo phase
        0, 0 # gates and LEDs, RX config
    ], dtype=np.uint16)}

def compare_csv(fname, sock, proc,
                 initial_bufs=np.zeros(fc.FLOCRA_BUFS, dtype=np.uint16),
                 latencies=np.zeros(fc.FLOCRA_BUFS, dtype=np.uint32),
                 self_ref=True # use the CSV source file as the reference file to compare the output with
                 ):

    source_csv = os.path.join("csvs", fname + ".csv")
    lc = fc.csv2bin(source_csv,
                    quick_start=False, initial_bufs=initial_bufs, latencies=latencies)
    data = np.array(lc, dtype=np.uint32)

    # run simulation
    rx_data, msgs = sc.command({'run_seq': data.tobytes()} , sock)

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sock)
    sock.close()
    proc.wait(1) # wait a short time for simulator to close

    # compare resultant CSV with the reference    
    if self_ref:
        rdata = np.loadtxt(source_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)
        sdata = np.loadtxt(flocra_sim_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)

        rdata[1:,0] -= rdata[1,0] # subtract off initial offset time
        sdata[1:,0] -= sdata[1,0] # subtract off initial offset time

        return rdata.tolist(), sdata.tolist()
    else:
        ref_csv = os.path.join("csvs", "ref_" + fname + ".csv")
        with open(ref_csv, "r") as ref:
            refl = ref.read().splitlines()
        with open(flocra_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml

def compare_dict(source_dict, ref_fname, sock, proc,                 
                 initial_bufs=np.zeros(fc.FLOCRA_BUFS, dtype=np.uint16),
                 latencies=np.zeros(fc.FLOCRA_BUFS, dtype=np.uint32),
                 ignore_start_delay=True
                 ):

    lc = fc.dict2bin(source_dict, initial_bufs=initial_bufs, latencies=latencies)
    data = np.array(lc, dtype=np.uint32)

    # run simulation
    rx_data, msgs = sc.command({'run_seq': data.tobytes()} , sock)

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sock)
    sock.close()
    proc.wait(1) # wait a short time for simulator to close

    ref_csv = os.path.join("csvs", ref_fname + ".csv")
    with open(ref_csv, "r") as ref:
        refl = ref.read().splitlines()
    with open(flocra_sim_csv, "r") as sim:
        siml = sim.read().splitlines()
    # return refl, siml

    ref_csv = os.path.join("csvs", ref_fname + ".csv")    
    if ignore_start_delay:
        rdata = np.loadtxt(ref_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)
        sdata = np.loadtxt(flocra_sim_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)

        rdata[1:,0] -= rdata[1,0] # subtract off initial offset time
        sdata[1:,0] -= sdata[1,0] # subtract off initial offset time

        return rdata.tolist(), sdata.tolist()
    else:
        with open(ref_csv, "r") as ref:
            refl = ref.read().splitlines()
        with open(flocra_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml

def compare_expt_dict(source_dict, ref_fname, sock, proc,
                      # initial_bufs=np.zeros(fc.FLOCRA_BUFS, dtype=np.uint16),
                      # latencies=np.zeros(fc.FLOCRA_BUFS, dtype=np.uint32),
                      ignore_start_delay=True,
                      **kwargs):
    """Arguments the same as for compare_dict(), except that the source
    dictionary is in floating-point units, and the kwargs are passed
    to the Experiment class constructor. Note that the initial_bufs
    and latencies are supplied to the Experiment class from the
    classes in grad_board.py.
    """

    e = exp.Experiment(prev_socket=sock, seq_dict=source_dict, **kwargs)

    # run simulation
    rx_data, msgs = e.run()

    # halt simulation
    sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), sock)
    sock.close()
    proc.wait(1) # wait a short time for simulator to close

    ref_csv = os.path.join("csvs", ref_fname + ".csv")
    with open(ref_csv, "r") as ref:
        refl = ref.read().splitlines()
    with open(flocra_sim_csv, "r") as sim:
        siml = sim.read().splitlines()
    # return refl, siml

    ref_csv = os.path.join("csvs", ref_fname + ".csv")    
    if ignore_start_delay:
        rdata = np.loadtxt(ref_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)
        sdata = np.loadtxt(flocra_sim_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)

        rdata[1:,0] -= rdata[1,0] # subtract off initial offset time
        sdata[1:,0] -= sdata[1,0] # subtract off initial offset time

        return rdata.tolist(), sdata.tolist()
    else:
        with open(ref_csv, "r") as ref:
            refl = ref.read().splitlines()
        with open(flocra_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml    

class ModelTest(unittest.TestCase):
    """Main test class for general HDL and compiler development/debugging;
    inputs to the API and UUT are either a CSV file or a dictionary,
    output is another CSV file, and comparison is either between the
    input and output CSV files (with allowance made for memory/startup
    delays) or between the output CSV and a reference CSV file. If a
    dictionary is used as input, a reference CSV file must be created.
    """

    @classmethod
    def setUpClass(cls):
        # TODO make this check for a file first
        os.system("make -j4 -s -C " + os.path.join(flocra_sim_path, "build"))
        os.system("fallocate -l 516KiB /tmp/marcos_server_mem")
        os.system("killall flocra_sim") # in case other instances were started earlier

        warnings.simplefilter("ignore", fc.FloServerWarning)
    
    def setUp(self):
        # start simulation
        if flocra_sim_fst_dump:
            self.p = subprocess.Popen([os.path.join(flocra_sim_path, "build", "flocra_sim"), "both", flocra_sim_csv, flocra_sim_fst],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT)
        else:
            self.p = subprocess.Popen([os.path.join(flocra_sim_path, "build", "flocra_sim"), "csv", flocra_sim_csv],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT)


        # open socket
        time.sleep(0.05) # give flocra_sim time to start up
        
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port)) # only connect to local simulator
        self.packet_idx = 0

    def tearDown(self):
        # self.p.terminate() # if not already terminated
        # self.p.kill() # if not already terminated
        self.s.close()

        if flocra_sim_fst_dump:
            # open GTKWave
            os.system("gtkwave " + flocra_sim_fst + " " + os.path.join(flocra_sim_path, "src", "flocra_sim.sav"))

    ## Tests are approximately in order of complexity

    def test_single(self):
        """ Basic state change on a single buffer """
        refl, siml = compare_csv("test_single", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_four_par(self):
        """ State change on four buffers in parallel """
        refl, siml = compare_csv("test_four_par", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_long_time(self):        
        """ State change on four buffers in parallel """
        max_orig = fc.COUNTER_MAX
        fc.COUNTER_MAX = 0xfff # temporarily reduce max time used by compiler
        refl, siml = compare_csv("test_long_time", self.s, self.p)
        fc.COUNTER_MAX = max_orig
        self.assertEqual(refl, siml)

    def test_single_quick(self):
        """ Quick successive state changes on a single buffer 1 cycle apart """
        refl, siml = compare_csv("test_single_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_single_delays(self):
        """ State changes on a single buffer with various delays in between"""
        refl, siml = compare_csv("test_single_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_quick(self):
        """ Quick successive state changes on two buffers, one cycle apart """
        refl, siml = compare_csv("test_two_quick", self.s, self.p)
        self.assertEqual(refl, siml)
    
    def test_two_delays(self):
        """ State changes on two buffers, various delays in between """
        refl, siml = compare_csv("test_two_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_three_quick(self):
        """ Quick successive state changes on two buffers, one cycle apart """
        refl, siml = compare_csv("test_three_quick", self.s, self.p)
        self.assertEqual(refl, siml)
    
    def test_three_delays(self):
        """ Successive state changes on three buffers, two cycles apart """
        refl, siml = compare_csv("test_three_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_mult_quick(self):
        """ Quick successive state changes on multiple buffers, 1 cycle apart """
        refl, siml = compare_csv("test_mult_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_many_quick(self):
        """ Many quick successive state changes on multiple buffers, all 1 cycle apart """
        refl, siml = compare_csv("test_many_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_stream_quick(self):
        """ Bursts of state changes on multiple buffers with uneven gaps for each individual buffer, each state change 1 cycle apart """
        refl, siml = compare_csv("test_stream_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_times(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
        refl, siml = compare_csv("test_uneven_times", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_sparse(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
        refl, siml = compare_csv("test_uneven_sparse", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_cfg(self):
        """ Configuration and LED bits/words """
        refl, siml = compare_csv("test_cfg", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_rx_simple(self):
        """ RX window with realistic RX rate configuration, resetting and gating """
        refl, siml = compare_csv("test_rx_simple", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_uneven_latencies(self):
        """Simultaneous state changes on two buffers, however 2nd buffer is
        specified to have 1 cycle of latency more than 1st - so its changes
        occur earlier to compensate"""
        refl, siml = compare_csv("test_two_uneven_latencies", self.s, self.p,
                                  latencies=np.array([
                                      0,0,0,0,
                                      0,0,1,0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      0], dtype=np.uint16),
                                  self_ref=False)
        self.assertEqual(refl, siml)

    def test_many_uneven_latencies(self):
        """Simultaneous state changes on four buffers, however they are
        assumed to all have different latencies relative to each other - thus
        out-of-sync change requests turn out in sync"""
        refl, siml = compare_csv("test_many_uneven_latencies", self.s, self.p,
                                  latencies=np.array([
                                      0, 0, 0, # grad
                                      0, 0, # rx
                                      2, 4, 6, 8, # tx
                                      0, 0, 0, 0, 0, 0, # lo phase
                                      0, 0 # gates and LEDs, RX config
                                  ], dtype=np.uint16),
                                  self_ref=False)
        self.assertEqual(refl, siml)
        
    def test_fhd_single(self):
        """Single state change on GPA-FHDO x gradient output, default SPI
        clock divisor; simultaneous change on TX0i"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csv("test_fhd_single", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_fhd_series(self):
        """Series of state changes on GPA-FHDO x gradient output, default SPI
        clock divisor """
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csv("test_fhd_series", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_fhd_multiple(self):
        """A few state changes on GPA-FHDO gradient outputs, default SPI clock divisor"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csv("test_fhd_multiple", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)

    def test_fhd_many(self):
        """Many state changes on GPA-FHDO gradient outputs, default SPI clock divisor - simultaneous with similar TX changes"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csv("test_fhd_many", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_fhd_too_fast(self):
        """Two state changes on GPA-FHDO gradient outputs, default SPI clock
        divisor - too fast for the divider, second state change won't
        be applied and server should notice the error
        """
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        
        # Run twice, to catch two different warnings (I couldn't find a more straightforward way to do this)
        with self.assertWarns( RuntimeWarning, msg="expected gpa-fhdo error not observed") as cmr:
            refl, siml = compare_csv("test_fhd_too_fast", self.s, self.p, self_ref=False, **fhd_config)
        with self.assertWarns( UserWarning, msg="expected flocompile warning not observed") as cmu:
            self.tearDown()
            self.setUp()
            refl, siml = compare_csv("test_fhd_too_fast", self.s, self.p, self_ref=False, **fhd_config)
            
        fc.grad_board = gb_orig
        # self.assertEqual( str(cm.exception) , "gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual( str(cmu.warning), "Gradient updates are too frequent for selected SPI divider. Missed samples are likely!")
        self.assertEqual( str(cmr.warning) , "ERROR: gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual(refl, siml)

    def test_oc1_single(self):
        """Single state change on ocra1 x gradient output, default SPI clock
        divisor; simultaneous change on TX0i
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csv("test_oc1_single", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)

    def test_oc1_series(self):
        """Series of state changes on ocra1 x gradient output, default SPI
        clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csv("test_oc1_series", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)        

    def test_oc1_two(self):
        """Two sets of simultaneous state changes on ocra1 gradient outputs,
        default SPI clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csv("test_oc1_two", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)        

    def test_oc1_four(self):
        """Four simultaneous state changes on ocra1 gradient outputs, default
        SPI clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csv("test_oc1_four", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)

    def test_oc1_many(self):
        """Multiple simultaneous state changes on ocra1 gradient outputs, default
        SPI clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csv("test_oc1_many", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_oc1_too_fast(self):
        """Two state changes on ocra1 gradient outputs, default SPI clock
        divisor - too fast for the divider, second state change won't
        be applied and server should notice the error
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        
        # Run twice, to catch two different warnings (I couldn't find a more straightforward way to do this)
        with self.assertWarns( RuntimeWarning, msg="expected ocra1 error not observed") as cmr:
            refl, siml = compare_csv("test_oc1_too_fast", self.s, self.p, self_ref=False, **oc1_config)
        with self.assertWarns( UserWarning, msg="expected flocompile warning not observed") as cmu:
            self.tearDown()
            self.setUp()
            refl, siml = compare_csv("test_oc1_too_fast", self.s, self.p, self_ref=False, **oc1_config)
            
        fc.grad_board = gb_orig
        # self.assertEqual( str(cm.exception) , "gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual( str(cmu.warning), "Gradient updates are too frequent for selected SPI divider. Missed samples are likely!")
        self.assertEqual( str(cmr.warning) , "ERROR: ocra1 gradient error; possibly missing samples")
        self.assertEqual(refl, siml)

    def test_single_dict(self):
        """ Basic state change on a single buffer. Dict version"""
        d = {'tx0_i': (np.array([100]), np.array([10000]))}
        refl, siml = compare_dict(d, "test_single", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_four_par_dict(self):
        """ State change on four buffers in parallel. Dict version"""
        d = {'tx0_i': (np.array([100]), np.array([5000])),
             'tx0_q': (np.array([100]), np.array([15000])),
             'tx1_i': (np.array([100]), np.array([25000])),
             'tx1_q': (np.array([100]), np.array([35000]))}
        refl, siml = compare_dict(d, "test_four_par", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_sparse_dict(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart. Dict version"""
        d = {'tx0_i': (np.array([100, 110, 113, 114, 115, 130, 152, 153, 156, 170, 174, 300, 10000]),
                       np.array([1, 5, 9, 13, 17, 25, 33, 37, 41, 45, 49, 53, 57])),
             'tx0_q': (np.array([110, 113, 114, 115, 116, 152, 156, 170, 174, 300, 10001]),
                       np.array([6, 10, 14, 18, 22, 34, 42, 46, 50, 54, 62])),
             'tx1_i': (np.array([113, 114, 115, 116, 130, 152, 153, 170, 177, 300, 10001]),
                       np.array([11, 15, 19, 23, 27, 35, 39, 47, 51, 55, 63])),
             'tx1_q': (np.array([114, 115, 116, 130, 150, 152, 156, 170, 177, 300, 10000]),
                       np.array([16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60])) }
        refl, siml = compare_dict(d, "test_uneven_sparse", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_cfg_dict(self):
        """ Configuration and LED bits/words. Dict version"""
        d = {'rx0_rate': ( np.array([100, 110, 208, 210]), np.array([1, 0, 1, 0]) ),
             'rx1_rate': ( np.array([110, 120, 205, 208]), np.array([1, 0, 1, 0]) ),
             'rx0_rate_valid': ( np.array([120, 130, 202, 205]), np.array([1, 0, 1, 0]) ),
             'rx1_rate_valid': ( np.array([130, 140, 199, 202]), np.array([1, 0, 1, 0]) ),
             'rx0_rst_n': ( np.array([140, 150, 196, 199]), np.array([1, 0, 1, 0]) ),
             'rx1_rst_n': ( np.array([150, 160, 194, 196]), np.array([1, 0, 1, 0]) ),
             'tx_gate': ( np.array([160, 170, 193, 194]), np.array([1, 0, 1, 0]) ),
             'rx_gate': ( np.array([170, 180, 192, 193]), np.array([1, 0, 1, 0]) ),
             'trig_out': ( np.array([180, 190, 191, 192]), np.array([1, 0, 1, 0]) ),
             'leds': ( np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 191, 192, 193, 194, 196, 199, 202, 205, 208, 210]),
                       np.array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 255]) )
             }
        refl, siml = compare_dict(d, "test_cfg", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_fhd_many_dict(self):
        """Many state changes on GPA-FHDO gradient outputs, default SPI clock divisor - simultaneous with similar TX changes. Dict version"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        d = {'tx0_i': (np.array([600, 1800]), np.array([1, 65532])),
             'tx0_q': (np.array([900, 2100]), np.array([2, 65533])),
             'tx1_i': (np.array([1200, 2400]), np.array([3, 65534])),
             'tx1_q': (np.array([1500, 2700]), np.array([4, 65535])),
             'fhdo_vx': (np.array([600, 1800]), np.array([1, 65532])),
             'fhdo_vy': (np.array([900, 2100]), np.array([2, 65533])),
             'fhdo_vz': (np.array([1200, 2400]), np.array([3, 65534])),
             'fhdo_vz2': (np.array([1500, 2700]), np.array([4, 65535]))}
        refl, siml = compare_dict(d, "test_fhd_many", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_single_expt(self):
        """ Basic state change on a single buffer. Experiment version"""
        d = {'tx0_i': (np.array([1]), np.array([0.5]))}
        expt_args = {'rx_t': 2, 'local_grad_board': 'ocra1'}
        refl, siml = compare_expt_dict(d, "test_single_expt", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

        # test for the GPA-FHDO too
        self.tearDown(); self.setUp()
        expt_args['local_grad_board'] = 'gpa-fhdo'
        refl, siml = compare_expt_dict(d, "test_single_expt", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

    def test_four_par_expt_iq(self):
        """ State change on four buffers in parallel. Experiment version using complex inputs"""
        d = {'tx0': (np.array([1]), np.array([0.5+0.2j])), 'tx1': (np.array([1]), np.array([-0.3+1j]))}
        expt_args = {'rx_t': 2, 'local_grad_board': 'ocra1'}
        refl, siml = compare_expt_dict(d, "test_four_par_expt_iq", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

        self.tearDown(); self.setUp()
        expt_args['local_grad_board'] = 'gpa-fhdo'
        refl, siml = compare_expt_dict(d, "test_four_par_expt_iq", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

    def test_uneven_sparse_ocra1_expt(self):
        """ Miscellaneous pulses on TX and gradients, with various acquisition windows """
        d = {'tx0': (np.array([10,15, 30,35, 100,105]), np.array([1,0, 0.8j,0, 0.7+0.2j,0])),
             'tx1': (np.array([5,20,  50,70,  110,125]), np.array([-1j,0,  -0.5j,0,  0.5+0.3j,0])),

             'grad_vx': (np.array([ 5, 30, 43, 50, 77]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vy': (np.array([10, 23, 43, 57, 84]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vz': (np.array([10, 23,     65, 77]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vz': (np.array([10,     43, 65, 77]), np.array([-1, 1, 0.5, -0.5, 0])),             

if __name__ == "__main__":
    unittest.main()        
