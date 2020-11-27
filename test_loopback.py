#!/usr/bin/env python3
#
# loopback test using ocra-pulseq
#

import numpy as np
import matplotlib.pyplot as plt
import pdb
import experiment as ex
from pulseq_assembler import PSAssembler
st = pdb.set_trace

if __name__ == "__main__":
    lo_freq = 2.1 # MHz
    tx_t = 0.994 # us
    rx_t = 0.497
    clk_t = 0.007
    grad_interval = 4 * tx_t # us between 4-channel updates
    ps = PSAssembler(rf_center=lo_freq*1e6,
                     # how many Hz the max amplitude of the RF will produce; i.e. smaller causes bigger RF V to compensate
                     rf_amp_max=16,
                     grad_max=10,
                     clk_t=clk_t,
                     tx_t=tx_t,
                     grad_t=grad_interval)
    _, _, cb, readout_samples = ps.assemble('../ocra-pulseq/test_files/test_loopback.seq')

    exp = ex.Experiment(samples=readout_samples, # TODO: get information from PSAssembler
                        lo_freq=lo_freq,
                        tx_t=tx_t,
                        rx_t=rx_t, # TODO: get information from PSAssembler
                        grad_channels=3,
                        grad_t=grad_interval/3,
                        assert_errors=False)
    
    exp.define_instructions(cb)
    x = np.linspace(0,2*np.pi, 100)
    ramp_sine = np.sin(2 * x)
    exp.add_tx(ps.tx_arr)
    exp.add_grad(ps.gr_arr)

    # plt.plot(ps.gr_arr[0]);plt.show()

    # for k in range(1):

    # exp.rx_div_real = 25 # temporary for testing
    
    data = exp.run() # Comment out this line to avoid running on the hardware

    plt.plot(data.imag)
    plt.show()

    # st()
