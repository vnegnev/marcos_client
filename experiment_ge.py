import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as sig

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz
from ocra_lib.assembler import Assembler
import server_comms as sc

from experiment import Experiment

def test_grad_echo_Elena():

    exp = Experiment(samples=600,
                     lo_freq=14.2375,
                     tx_t=0.1,
                     rx_t=2,
                     instruction_file="ocra_lib/grad_echo_elena.txt")

     # RF pulse
    t = np.linspace(0, 20, 201) # goes to 20us, samples every 10ns; length of pulse must be adjusted in grad_echo_elena.txt

    # sinc pulse
    tx_x = np.sinc( (t - 10) / 6 )
    tx_idx = exp.add_tx(tx_x) # add the data to the ocra TX memory


    # gradient echo; 190 samples total: 50 for first ramp, 140 for second ramp
    grad = np.hstack([
        np.linspace(0, 0.9, 10), np.ones(30), np.linspace(0.9, 0, 10), # first ramp up/down
        np.linspace(0,-0.285, 20), -0.3 * np.ones(100), np.linspace(-0.285, 0, 20)
        ])

    # Correct for DC offset and scaling
    scale = 0.9
    offset = 0.0
    grad_corr = grad*scale + offset

    grad_idx = exp.add_grad(grad_corr, grad_corr, grad_corr)
    if False: # set to true if you want to plot the x gradient waveform
        plt.plot(grad_corr);plt.show()

    data = exp.run()

    plt.plot(np.real(data))
    plt.plot(np.imag(data))
    plt.plot(np.abs(data))
    plt.legend(['real', 'imag', 'abs'])
    plt.show()

if __name__ == "__main__":
    test_grad_echo_Elena()