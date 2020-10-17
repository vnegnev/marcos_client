#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import experiment as ex
from pulseq_assembler import PSAssembler

def test_grad_echo_loop():
    exp = ex.Experiment(samples=1900 + 210,
                        lo_freq=0.5,
                        grad_channels=3,
                        instruction_file='ocra_lib/grad_echo.txt',
                        grad_t=0.8,
                        print_infos=True)
    
    # RF pulse
    t = np.linspace(0, 200, 2001) # goes to 200us, samples every 100ns; length of pulse must be adjusted in grad_echo.txt

    if False:
        # square pulse at an offset frequency
        freq = 0.1 # MHz, offset from LO freq (DC up to a few MHz possible)
        tx_x = np.cos(2*np.pi*freq*t) + 1j*np.sin(2*np.pi*freq*t) # I,Q samples
        tx_idx = exp.add_tx(tx_x) # add the data to the ocra TX memory
    else:
        # sinc pulse
        tx_x = np.sinc( (t - 100) / 25 )
        tx_idx = exp.add_tx(tx_x) # add the data to the ocra TX memory

    # 2nd RF pulse, for testing
    tx_x2 = tx_x*0.5
    tx_idx2 = exp.add_tx(tx_x2)

    # gradient echo; 190 samples total: 50 for first ramp, 140 for second ramp
    grad = np.hstack([
        np.linspace(0, 0.9, 10), np.ones(30), np.linspace(0.9, 0, 10), # first ramp up/down
        np.linspace(0,-0.285, 20), -0.3 * np.ones(100), np.linspace(-0.285, 0, 20)
        ])

    for k in np.linspace(-1, 1, 101):
    # Correct for DC offset and scaling
        scale = k
        offset = 0
        grad_corr = grad*scale + offset

        exp.clear_grad()
        grad_idx = exp.add_grad([grad_corr, grad_corr, grad_corr])
        if False: # set to true if you want to plot the x gradient waveform
            plt.plot(grad_corr);plt.show()

        data = exp.run()
        # import time
        # time.sleep(0.1)

        if False:
            plt.plot(np.real(data))
            plt.plot(np.imag(data))    
            plt.show()

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('test_grad_echo_loop()')
    test_grad_echo_loop()
