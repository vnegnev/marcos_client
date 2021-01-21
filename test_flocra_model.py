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
# python -m unittest test_flocompile.CsvTest.test_many_quick

import sys, os, subprocess, warnings, socket, unittest, time
import numpy as np
import matplotlib.pyplot as plt

import server_comms as sc

import flocompile as fc

import pdb
st = pdb.set_trace

ip_address = "localhost"
port = 11111
flocra_path = os.path.join("..", "flocra")
flocra_sim_csv = os.path.join("/tmp", "flocra_sim.csv")

def compare_csvs(fname, sock, proc,
                 initial_bufs=np.zeros(16, dtype=np.uint16),
                 latencies=np.zeros(16, dtype=np.uint32),
                 diff_times=True
                 ):

    lc = fc.csv2bin(os.path.join("csvs", fname + ".csv"),
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
    ref_csv = os.path.join("csvs", "ref_" + fname + ".csv")
    
    if diff_times:
        rdata = np.loadtxt(ref_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)
        sdata = np.loadtxt(flocra_sim_csv, skiprows=1, delimiter=',', comments='#').astype(np.uint32)

        rdata[1:,0] -= rdata[1,0] # subtract off initial offset time
        sdata[1:,0] -= sdata[1,0] # subtract off initial offset time

        return rdata.tolist(), sdata.tolist()
    else:
        with open(refpath, "r") as ref:
            refl = ref.read().splitlines()
        with open(flocra_sim_csv, "r") as sim:
            siml = sim.read().splitlines()
        return refl, siml

class CsvTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO make this check for a file first
        os.system("make -j4 -s -C " + os.path.join(flocra_path, "build"))
        os.system("fallocate -l 516KiB /tmp/marcos_server_mem")
        os.system("killall flocra_sim") # in case other instances were started earlier
    
    def setUp(self):
        # start simulation
        self.p = subprocess.Popen([os.path.join(flocra_path, "build", "flocra_sim"), "csv", flocra_sim_csv],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.STDOUT)

        ## Uncomment to produce an FST dump for debugging with GTKWave, in the case of a particular misbehaving test
        # self.p = subprocess.Popen([os.path.join(flocra_path, "build", "flocra_sim"), "fst", "/tmp/flocra_sim.fst"])

        # open socket
        time.sleep(0.05) # give flocra_sim time to start up
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port)) # only connect to local simulator
        self.packet_idx = 0

    def tearDown(self):
        # self.p.terminate() # if not already terminated
        # self.p.kill() # if not already terminated
        self.s.close()

    ## Tests are approximately in order of complexity

    # def test_single(self):
    #     """ Basic state change on a single buffer """
    #     refl, siml = compare_csvs("test_single", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test_four_par(self):
    #     """ State change on four buffers in parallel """
    #     refl, siml = compare_csvs("test_four_par", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test_single_quick(self):
    #     """ Quick successive state changes on a single buffer 1 cycle apart """
    #     refl, siml = compare_csvs("test_single_quick", self.s, self.p)
    #     self.assertEqual(refl, siml)

    def test_single_delays(self):
        """ State changes on a single buffer with various delays in between"""
        refl, siml = compare_csvs("test_single_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    # def test_two_quick(self):
    #     """ Quick successive state changes on two buffers, 2 cycles apart """
    #     refl, siml = compare_csvs("test_two_quick", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test_mult_quick(self):
    #     """ Quick successive state changes on multiple buffers, 1 cycle apart """
    #     refl, siml = compare_csvs("test_mult_quick", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test_many_quick(self):
    #     """ Many quick successive state changes on multiple buffers, all 1 cycle apart """
    #     refl, siml = compare_csvs("test_many_quick", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test_stream_quick(self):
    #     """ Bursts of state changes on multiple buffers with uneven gaps for each individual buffer, each state change 1 cycle apart """
    #     refl, siml = compare_csvs("test_stream_quick", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test_uneven_times(self):
    #     """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
    #     refl, siml = compare_csvs("test_uneven_times", self.s, self.p)
    #     self.assertEqual(refl, siml)    

    # @unittest.expectedFailure        
    # def test06_nolat(self):
    #     """ Simultaneous state change on RX and TX, unmatched latency """
    #     refl, siml = compare_csvs("test06", self.s, self.p)
    #     self.assertEqual(refl, siml)

    # def test07_lat(self):
    #     """ Simultaneous state change on RX and TX, matched latency """
    #     mlat = np.zeros(16, dtype=np.uint16)
    #     # mlat = np.ones(16, dtype=np.uint16)
    #     mlat[5:9] = np.ones(4)
    #     refl, siml = compare_csvs("test07", self.s, self.p,
    #                               latencies=mlat)
    #     self.assertEqual(refl, siml)

if __name__ == "__main__":
    unittest.main()
