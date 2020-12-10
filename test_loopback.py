#!/usr/bin/env python3
#
# loopback test using ocra-pulseq
#

import numpy as np
import matplotlib.pyplot as plt
import pdb, sys
import experiment as ex
from pulseq_assembler import PSAssembler
st = pdb.set_trace

if __name__ == "__main__":
    lo_freq = 2.1 # MHz
    tx_t = 0.994 # us
    clk_t = 0.007
    grad_channels = 3
    grad_interval = 9.996 # us between updates, i.e. grad raster time

    do_single_test = True
    do_jitter_test = False

    if len(sys.argv) > 1 and "jitter" in sys.argv:
        do_jitter_test = True
        do_single_test = False
    
    ps = PSAssembler(rf_center=lo_freq*1e6,
                     # how many Hz the max amplitude of the RF will produce; i.e. smaller causes bigger RF V to compensate
                     rf_amp_max=16,
                     grad_max=10,
                     clk_t=clk_t,
                     tx_t=tx_t,
                     grad_t=grad_interval,
                     grad_pad=2,
                     addresses_per_grad_sample=3)
    tx_arr, grad_arr, cb, params = ps.assemble('../ocra-pulseq/test_files/test_loopback.seq', byte_format=False)
    
    exp = ex.Experiment(samples=params['readout_number'], # TODO: get information from PSAssembler
                        lo_freq=lo_freq,
                        tx_t=tx_t,
                        rx_t=params['rx_t'],
                        grad_channels=grad_channels,
                        grad_t=grad_interval/grad_channels,
                        assert_errors=False,
                        print_infos=True)
    
    exp.define_instructions(cb)
    x = np.linspace(0,2*np.pi, 100)
    ramp_sine = np.sin(2 * x)
    exp.add_tx(tx_arr)
    exp.add_grad(grad_arr)

    # plt.plot(ps.gr_arr[0]);plt.show()

    # for k in range(1):

    # exp.rx_div_real = 25 # temporary for testing

    if do_single_test:
        exp.run()
        
    if do_jitter_test:
        data = []
        trials = 1000
        for k in range(trials):
            data.append( exp.run() ) # Comment out this line to avoid running on the hardware
        
        taxis = np.arange(params['readout_number'])*params['rx_t']
        plt.figure(figsize=(10,9))

        good_data = []
        bad_data = []

        for d in data:
            if np.abs(d[-1]) == 0.0:
                bad_data.append(d)
            else:
                good_data.append(d)

        lgd = len(good_data)
        lbd = len(bad_data)
                
        plt.subplot(2,1,1)
        for d in good_data:
            plt.plot(taxis, d.real )
        plt.ylabel('loopback rx amplitude')
        plt.title('passing loopback data ({:d}/{:d}, {:.2f}%)'.format(lgd, lgd+lbd, 100*lgd/(lgd+lbd)))
        plt.grid(True)

        plt.subplot(2,1,2)
        for d in bad_data:
            plt.plot(taxis, d.real )
        plt.ylabel('loopback rx amplitude')
        plt.title('failing loopback data ({:d}/{:d}, {:.2f}%)'.format(lbd, lgd+lbd, 100*lbd/(lgd+lbd)))
        plt.grid(True)        
        
        plt.show()
