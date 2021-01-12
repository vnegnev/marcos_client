#!/usr/bin/env python3
#
# To run a single test, use e.g.:
# python -m unittest test_flocompile.CsvTest.test0

import sys, os, subprocess, warnings, socket, unittest, time
import numpy as np
import matplotlib.pyplot as plt

import server_comms as sc

import flocompile as fc

import pdb
st = pdb.set_trace

ip_address = "localhost"
port = 11111
flocra_path = "../flocra"

class CsvTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO make this check for a file first
        os.system("make -j4 -s -C " + os.path.join(flocra_path, "build"))
        os.system("fallocate -l 516KiB /tmp/marcos_server_mem")
        os.system("killall flocra_sim") # in case other instances were started earlier
    
    def setUp(self):
        # start simulation
        os.system(os.path.join(flocra_path, "build", "flocra_sim") + " csv /tmp/flocra_sim.csv &")
        # open socket
        time.sleep(0.05) # give flocra_sim time to start up
        self.csvf = open("/tmp/flocra_sim.csv", 'r')
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port)) # only connect to local simulator
        self.packet_idx = 0        

    def tearDown(self):
        # halt simulation
        sc.send_packet(sc.construct_packet({}, 0, command=sc.close_server_pkt), self.s)
        # close socket
        self.s.close()

    def test01(self):
        """ Linear increase on one channel, in time and value"""
        lc = fc.csv2bin("test_csvs/test01.csv")
        lc.append(fc.insta(fc.IFINISH, 0))
        data = np.array(lc, dtype=np.uint32)
        rx_data, msgs = sc.command({'run_seq': data.tobytes()} , self.s)

        st()

if __name__ == "__main__":
    unittest.main()
