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
    lo_freq = 5 # MHz
    tx_t = 1 # us
    grad_interval = 5 # us between 4-channel updates
    ps = PSAssembler(rf_center=lo_freq*1e6,
                     rf_amp_max=5e3,
                     grad_max=10,
                     clk_t=0.1,
                     tx_t=tx_t,
                     grad_t=grad_interval)
    _, _, cb = ps.assemble('../ocra-pulseq/test_files/test4.seq')
    tx_arr = ps.tx_arr
    gr_arr = ps.gr_arr

    exp = ex.Experiment(samples=1000, # TODO: get information from PSAssembler
                        lo_freq=lo_freq,
                        tx_t=tx_t,
                        rx_t=1, # TODO: get information from PSAssembler
                        grad_channels=3,
                        grad_t=grad_interval/3)

    exp.define_instructions(cb)
    exp.add_tx(tx_arr)
    exp.add_grad(gr_arr)
    data = exp.run()

