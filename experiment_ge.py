import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.io as io
import scipy.signal as sig
import math

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz
from ocra_lib.assembler import Assembler
import server_comms as sc

from experiment import Experiment

def sinc(x,tx_time,Nlobes,alpha):
    y = []
    t0 = (tx_time/2) / Nlobes
    for ii in x:
        if ii==0.0:
            yy = 1.0
        else:
            yy = t0* ((1 - alpha) + alpha*np.cos(ii/ Nlobes / t0))*math.sin(ii/t0)/ii
        y = np.append(y, yy)
    return y

def test_grad_echo_Elena():
    samples_tmp = 140
    tx_dt = 0.1
    rx_dt = 3.33 #desired sampling dt
    rx_dt_corr = rx_dt *0.5 #correction till the bug is fixed
    
    exp = Experiment(samples=samples_tmp,
                     lo_freq= 14.1711,
                     tx_t= tx_dt,
                     rx_t=rx_dt_corr,
                     instruction_file="ocra_lib/grad_echo_elena.txt")

     # RF pulse`
    tx_time = 34
    t = np.linspace(0, tx_time, math.ceil(tx_time/tx_dt)+1) # goes to tx_time us, samples every tx_t us; length of pulse must be adjusted in grad_echo_elena.txt
    
    alpha = 0.46 # alpha=0.46 for Hamming window, alpha=0.5 for Hanning window
    Nlobes = 4
    ampl = 0.66 #0.8 in radioprocessor

    #sinc pulse with Hamming window
    tx_x = ampl * sinc(math.pi*(t - tx_time/2),tx_time,Nlobes,alpha)
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
    data_mV = data*1000
    
    # time vector for representign the received data
    samples_data = len(data)
    t_rx = np.linspace(0, rx_dt*samples_data, samples_data) #us

    io.savemat('test_rp_60.mat', dict(t_rx=t_rx, data_mV=data_mV))

    plt.plot(t_rx,np.real(data_mV))
    plt.plot(t_rx,np.imag(data_mV))
    plt.plot(t_rx,np.abs(data_mV))
    plt.legend(['real', 'imag', 'abs'])
    plt.xlabel('time (us)')
    plt.ylabel('signal received (mV)')
    plt.title('sampled data = %i' %samples_data)
    plt.show()

if __name__ == "__main__":
    test_grad_echo_Elena()