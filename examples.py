#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import experiment as ex

import pdb
st = pdb.set_trace

def radial():
    ## All times are relative to a single TR, starting at time 0
    lo_freq = 2 # MHz
    rf_amp = 0.5 # 1 = full-scale
    
    G = 0.2 # Gx = G cos (t), Gy = G sin (t)
    trs = 36 # TR repetitions
    angles = np.linspace(0, 2*np.pi, trs) # angle
    
    grad_tstart = 0 # us
    rf_tstart = 5 # us
    rf_tend = 10 # us
    rx_tstart = 20 # us
    rx_tend = 100 # us
    rx_period = 3 # us
    tr_total_time = 110 # start-finish TR time

    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends

    def radial_tr(tstart, th):
        gx = G * np.cos(th)
        gy = G * np.sin(th)
        dbg_sc = 1 # set to 0 to avoid 2nd RF debugging pulse
        value_dict = {
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,    rx_tstart + 15, rx_tend - 15]),
                     np.array([rf_amp, 0,    dbg_sc * (gx+gy*1j), 0]) ),
            'grad_vx': ( np.array([grad_tstart]),
                         np.array([gx]) ),
            'grad_vy': ( np.array([grad_tstart]),
                         np.array([gy]) ),            
            'rx0_rst_n' : ( np.array([rx_tstart, rx_tend]),
                            np.array([1, 0]) ),
            'tx_gate' : ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]),
                          np.array([1, 0]) )
            }
        
        for k, v in value_dict.items():
            # v for read, value_dict[k] for write
            value_dict[k] = (v[0] + tstart, v[1])

        return value_dict
    
    expt = ex.Experiment(lo_freq=lo_freq,
                         rx_t=rx_period)

    tr_t = 20 # start the first TR at 20us
    for th in angles:
        expt.add_flodict( radial_tr( tr_t, th ) )
        tr_t += tr_total_time

    rxd, msgs = expt.run()
    expt.close_server()

    rxr = rxd[4]['run_seq']
    plt.plot( np.array(rxr['rx0_i'], dtype=np.int32) )
    plt.plot( np.array(rxr['rx0_q'], dtype=np.int32) )
    plt.show()

def test_grad_echo_loop():
    exp = ex.Experiment(samples=1900 + 210,
                        lo_freq=0.5,
                        grad_channels=3,
                        instruction_file='ocra_lib/grad_echo.txt',
                        grad_t=0.8,
                        print_infos=True,
                        assert_errors=False)
    
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
    # test_grad_echo_loop()
    radial()
