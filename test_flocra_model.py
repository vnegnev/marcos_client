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
# python -m unittest test_flocra_model.CsvTest.test_many_quick

import sys, os, subprocess, warnings, socket, unittest, time
import numpy as np
import matplotlib.pyplot as plt

import server_comms as sc

import flocompile as fc

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

# Arguments for compare_csvs when running gradient tests
fhd_config = {
    'initial_bufs': np.array([
        # see flocra.sv, gradient control lines (lines 186-190, 05.02.2021)
        # strobe for both LSB and LSB, reset_n = 1, spi div = 10, grad board select (1 = ocra1, 2 = gpa-fhdo)
        (1 << 9) | (1 << 8) | (10 << 2) | 2,
        0, 0,
        0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0], dtype=np.uint16),
    'latencies': np.array([
        0, 276, 276, # grad latencies match SPI div
        0, 0, # rx
        0, 0, 0, 0, # tx
        0, 0, 0, 0, 0, 0, # lo phase
        0 # gates and LEDs
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
        0], dtype=np.uint16),
    'latencies': np.array([
        0, 268, 268, # grad latencies match SPI div
        0, 0, # rx
        0, 0, 0, 0, # tx
        0, 0, 0, 0, 0, 0, # lo phase
        0 # gates and LEDs
    ], dtype=np.uint16)}

def compare_csvs(fname, sock, proc,
                 initial_bufs=np.zeros(16, dtype=np.uint16),
                 latencies=np.zeros(16, dtype=np.uint32),
                 self_ref=True # use the CSV source file as the reference file to compare the output with
                 ):

    source_csv = os.path.join("csvs", fname + ".csv")
    lc = fc.csv2bin(source_csv,
                    quick_start=False, min_grad_clocks=200,
                    initial_bufs=initial_bufs,
                    latencies=latencies)

    lc.append(fc.insta(fc.IFINISH, 0))
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

class CsvTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO make this check for a file first
        os.system("make -j4 -s -C " + os.path.join(flocra_sim_path, "build"))
        os.system("fallocate -l 516KiB /tmp/marcos_server_mem")
        os.system("killall flocra_sim") # in case other instances were started earlier
    
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
        refl, siml = compare_csvs("test_single", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_four_par(self):
        """ State change on four buffers in parallel """
        refl, siml = compare_csvs("test_four_par", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_long_time(self):        
        """ State change on four buffers in parallel """
        max_orig = fc.COUNTER_MAX
        fc.COUNTER_MAX = 0xfff # temporarily reduce max time used by compiler
        refl, siml = compare_csvs("test_long_time", self.s, self.p)
        fc.COUNTER_MAX = max_orig
        self.assertEqual(refl, siml)

    def test_single_quick(self):
        """ Quick successive state changes on a single buffer 1 cycle apart """
        refl, siml = compare_csvs("test_single_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_single_delays(self):
        """ State changes on a single buffer with various delays in between"""
        refl, siml = compare_csvs("test_single_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_quick(self):
        """ Quick successive state changes on two buffers, one cycle apart """
        refl, siml = compare_csvs("test_two_quick", self.s, self.p)
        self.assertEqual(refl, siml)
    
    def test_two_delays(self):
        """ State changes on two buffers, various delays in between """
        refl, siml = compare_csvs("test_two_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_three_quick(self):
        """ Quick successive state changes on two buffers, one cycle apart """
        refl, siml = compare_csvs("test_three_quick", self.s, self.p)
        self.assertEqual(refl, siml)
    
    def test_three_delays(self):
        """ Successive state changes on three buffers, two cycles apart """
        refl, siml = compare_csvs("test_three_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_mult_quick(self):
        """ Quick successive state changes on multiple buffers, 1 cycle apart """
        refl, siml = compare_csvs("test_mult_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_many_quick(self):
        """ Many quick successive state changes on multiple buffers, all 1 cycle apart """
        refl, siml = compare_csvs("test_many_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_stream_quick(self):
        """ Bursts of state changes on multiple buffers with uneven gaps for each individual buffer, each state change 1 cycle apart """
        refl, siml = compare_csvs("test_stream_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_times(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
        refl, siml = compare_csvs("test_uneven_times", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_sparse(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
        refl, siml = compare_csvs("test_uneven_sparse", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_cfg(self):
        """ Configuration and LED bits/words """
        refl, siml = compare_csvs("test_cfg", self.s, self.p)
        self.assertEqual(refl, siml)
        
    def test_rx_simple(self):
        """ RX window with realistic RX rate configuration, resetting and gating """
        refl, siml = compare_csvs("test_rx_simple", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_uneven_latencies(self):
        """Simultaneous state changes on two buffers, however 2nd buffer is
        specified to have 1 cycle of latency more than 1st - so its changes
        occur earlier to compensate"""
        refl, siml = compare_csvs("test_two_uneven_latencies", self.s, self.p,
                                  latencies=np.array([
                                      0,0,0,0,
                                      0,0,1,0,
                                      0,0,0,0,
                                      0,0,0,0], dtype=np.uint16),
                                  self_ref=False)
        self.assertEqual(refl, siml)

    def test_many_uneven_latencies(self):
        """Simultaneous state changes on four buffers, however they are
        assumed to all have different latencies relative to each other - thus
        out-of-sync change requests turn out in sync"""
        refl, siml = compare_csvs("test_many_uneven_latencies", self.s, self.p,
                                  latencies=np.array([
                                      0, 0, 0, # grad
                                      0, 0, # rx
                                      2, 4, 6, 8, # tx
                                      0, 0, 0, 0, 0, 0, # lo phase
                                      0 # gates and LEDs
                                  ], dtype=np.uint16),
                                  self_ref=False)
        self.assertEqual(refl, siml)
        
    def test_fhd_single(self):
        """Single state change on GPA-FHDO x gradient output, default SPI
        clock divisor; simultaneous change on TX0i"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csvs("test_fhd_single", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_fhd_series(self):
        """Series of state changes on GPA-FHDO x gradient output, default SPI
        clock divisor """
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csvs("test_fhd_series", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_fhd_multiple(self):
        """A few state changes on GPA-FHDO gradient outputs, default SPI clock divisor"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csvs("test_fhd_multiple", self.s, self.p, **fhd_config)        
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)
        
    def test_fhd_many(self):
        """Many state changes on GPA-FHDO gradient outputs, default SPI clock divisor - simultaneous with similar TX changes"""
        gb_orig = fc.grad_board
        fc.grad_board = "gpa-fhdo"
        refl, siml = compare_csvs("test_fhd_many", self.s, self.p, **fhd_config)        
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
            refl, siml = compare_csvs("test_fhd_too_fast", self.s, self.p, self_ref=False, **fhd_config)
        with self.assertWarns( UserWarning, msg="expected flocompile warning not observed") as cmu:
            self.tearDown()
            self.setUp()
            refl, siml = compare_csvs("test_fhd_too_fast", self.s, self.p, self_ref=False, **fhd_config)
            
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
        refl, siml = compare_csvs("test_oc1_single", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)

    def test_oc1_series(self):
        """Series of state changes on ocra1 x gradient output, default SPI
        clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csvs("test_oc1_series", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)        

    def test_oc1_two(self):
        """Two sets of simultaneous state changes on ocra1 gradient outputs,
        default SPI clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csvs("test_oc1_two", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)        

    def test_oc1_four(self):
        """Four simultaneous state changes on ocra1 gradient outputs, default
        SPI clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csvs("test_oc1_four", self.s, self.p, **oc1_config)
        fc.grad_board = gb_orig
        self.assertEqual(refl, siml)

    def test_oc1_many(self):
        """Multiple simultaneous state changes on ocra1 gradient outputs, default
        SPI clock divisor
        """
        gb_orig = fc.grad_board
        fc.grad_board = "ocra1"
        refl, siml = compare_csvs("test_oc1_many", self.s, self.p, **oc1_config)
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
            refl, siml = compare_csvs("test_oc1_too_fast", self.s, self.p, self_ref=False, **oc1_config)
        with self.assertWarns( UserWarning, msg="expected flocompile warning not observed") as cmu:
            self.tearDown()
            self.setUp()
            refl, siml = compare_csvs("test_oc1_too_fast", self.s, self.p, self_ref=False, **oc1_config)
            
        fc.grad_board = gb_orig
        # self.assertEqual( str(cm.exception) , "gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual( str(cmu.warning), "Gradient updates are too frequent for selected SPI divider. Missed samples are likely!")
        self.assertEqual( str(cmr.warning) , "ERROR: ocra1 gradient error; possibly missing samples")
        self.assertEqual(refl, siml)

if __name__ == "__main__":
    unittest.main()
