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
    tx_t = 1.001 # us
    rx_t = 0.497
    clk_t = 0.007
    num_grad_channels = 3
    grad_interval = 10.003 # us between [num_grad_channels] channel updates

    gamma = 42570000 # Hz/T

    # value for tabletopMRI  gradient coil
    B_per_m_per_current = 0.02 # T/m/A, approximate value for tabletop gradient coil

    # values for gpa fhdo
    dac_voltage_per_current = 1/3.75 # V/A, theoretical value for gpa fhdo without 2.5 V offset!
    max_dac_voltage = 2.5

    # values for ocra1
    # max_dac_voltage = 5
    # dac_voltage_per_current = 0.2 # fill in value of gradient power amplifier here!

    max_Hz_per_m = max_dac_voltage / dac_voltage_per_current * B_per_m_per_current * gamma	
    # grad_max = max_Hz_per_m # factor used to normalize gradient amplitude, should be max value of the gpa used!	
    grad_max = 16 # unrealistic value used only for loopback test

    rf_amp_max = 10 # factor used to normalize RF amplitude, should be max value of system used!
    ps = PSAssembler(rf_center=lo_freq*1e6,
        # how many Hz the max amplitude of the RF will produce; i.e. smaller causes bigger RF V to compensate
        rf_amp_max=rf_amp_max,
        grad_max=grad_max,
        clk_t=clk_t,
        tx_t=tx_t,
        grad_t=grad_interval)
    _, _, cb, readout_samples = ps.assemble('../ocra-pulseq/test_files/test_loopback.seq')

    exp = ex.Experiment(samples=readout_samples, 
        lo_freq=lo_freq,
        tx_t=tx_t,
        rx_t=rx_t, # TODO: get information from PSAssembler
        grad_channels=num_grad_channels,
        grad_t=grad_interval/num_grad_channels,
        assert_errors=False)
    exp.define_instructions(cb)
    x = np.linspace(0,2*np.pi, 100)
    ramp_sine = np.sin(2*x)
    exp.add_tx(ps.tx_arr)
    exp.add_grad(ps.gr_arr)

    # plt.plot(ps.gr_arr[0]);plt.show()

    # for k in range(1):

    # exp.rx_div_real = 25 # temporary for testing

    exp.init_gpa()
    exp.calibrate_gpa_fhdo()

    data = exp.run() # Comment out this line to avoid running on the hardware

    plt.plot(data.imag)
    plt.show()

    # st()
