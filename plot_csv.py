#!/usr/bin/env python3
# Basic script to plot the CSV that flocra_sim produces, so that you can visualise the expected pulse sequence from the hardware.

import numpy as np
import matplotlib.pyplot as plt
import sys, pdb
st = pdb.set_trace

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python {:s} <csv_file.csv>".format(sys.argv[0]) )
        print("\t Example: python {:s} /tmp/flocra_sim.csv".format(sys.argv[0]) )
        exit()

    # data = np.genfromtxt(sys.argv[1],
    #                      dtype=None,
    #                      comments='#',
    #                      delimiter=',',
    #                      skip_header=0,
    #                      usecols=(0,1),
    #                      names=True)
    grad_board = "ocra1"

    data = np.loadtxt(sys.argv[1], skiprows=1, delimiter=',')
    data[1:, 0] = data[1:, 0] - data[1, 0] + 1 # remove dead time in the beginning taken up by simulated memory writes
    
    time_us = data[:,0]/122.88
    tx = data[:,1:5].astype(np.int16) / 32768
    fhdo = data[:,5:9].astype(np.int16) / 32768
    ocra1 = ( (data[:,9:13].astype(np.int32) ^ (1 << 17)) - (1 << 17) ).astype(np.int32) / 131072
    rx = data[:,14:19].astype(np.uint8)
    rx_en = rx[:, 4:] # ignore the rate logic, only plot the RX enables
    io = data[:,19:].astype(np.uint8)    

    fig, (txs, grads, rxs, ios) = plt.subplots(4, 1, figsize=(12,8), sharex='col')

    txs.step(time_us, tx, where='post')
    txs.legend(['tx0 i', 'tx0 q', 'tx1 i', 'tx1 q'])
    txs.grid(True)

    if grad_board == "ocra1":
        gdata = ocra1
        glegends = ['ocra1 x', 'ocra1 y', 'ocra1 z', 'ocra1 z2']
    elif grad_board == "gpa-fhdo":        
        gdata = fhdo
        glegends = ['fhdo x', 'fhdo y', 'fhdo z', 'fhdo z2']

    grads.step(time_us, gdata, where='post')
    grads.legend(glegends)

    rxs.step(time_us, rx_en, where='post')
    rxs.legend(["rx0 rstn", "rx1 rstn"])

    ios.step(time_us, io, where='post')
    ios.legend(['tx gate', 'rx gate', 'trig out', 'leds'])        
    grads.set_xlabel('time (us)')
    
    fig.tight_layout()
    plt.show()
    # st()
    
