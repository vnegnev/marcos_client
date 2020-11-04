#!/usr/bin/env python3
#
# ocra-pulseq test
#

import numpy as np
import matplotlib.pyplot as plt
import pdb
import experiment as ex
from pulseq_assembler import PSAssembler
st = pdb.set_trace

# def asm_parse(asm_file):
#     lines = []

#     for line in asm_file:
#         st()

if __name__ == "__main__":
    lo_freq = 2.1 # MHz
    # tx_t = 0.994 # us
    tx_t = 0.994 # us
    # rx_t = tx_t*5 # us
    # rx_t = 0.595
    rx_t = 0.497
    clk_t = 0.007
    grad_interval = tx_t # us between 4-channel updates
    ps = PSAssembler(rf_center=lo_freq*1e6,
                     # how many Hz the max amplitude of the RF will produce; i.e. smaller causes bigger RF V to compensate
                     # rf_amp_max=5e3,
                     # rf_amp_max=2e4,
                     # rf_amp_max=1e-4,
                     rf_amp_max=16,
                     grad_max=10,
                     clk_t=clk_t,
                     tx_t=tx_t,
                     grad_t=grad_interval)
    _, _, cb = ps.assemble('../ocra-pulseq/test_files/test2_mod2.seq')


    exp = ex.Experiment(samples=1000, # TODO: get information from PSAssembler
                        lo_freq=lo_freq,
                        tx_t=tx_t,
                        rx_t=rx_t, # TODO: get information from PSAssembler
                        grad_channels=3,
                        grad_t=grad_interval/3,
                        assert_errors=False)
    
    exp.define_instructions(cb)
    x = np.linspace(0,2*np.pi, 100)
    ramp_sine = np.sin(2 * x)
    exp.add_tx(ps.tx_arr) # * ramp_sine)
    # st()
    # grad_test = np.cos(np.linspace(0, 100, 2000));
    # exp.add_grad(ps.gr_arr)
    # ps.gr_arr = np.cos(np.linspace(0, 10, 100));
    # exp.add_grad([grad_test, grad_test, grad_test])

    # plt.plot(ps.gr_arr[2]);plt.show()

    # for k in range(1):

    # exp.rx_div_real = 25 # temporary for testing
    
    data = exp.run() # Comment out this line to avoid running on the hardware

    # plt.plot(data.imag)
    # plt.show()

    # st()
