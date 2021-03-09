#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import experiment as ex

import pdb
st = pdb.set_trace

def trapezoid(plateau_a, total_t, ramp_t, ramp_pts, total_t_end_to_end=True, base_a=0):
    """Helper function that just generates a Numpy array starting at time
    0 and ramping down at time total_t, containing a trapezoid going from a
    level base_a to plateau_a, with a rising ramp of duration ramp_t and
    sampling period ramp_ts."""

    # ramp_pts = int( np.ceil(ramp_t/ramp_ts) ) + 1
    rise_ramp_times = np.linspace(0, ramp_t, ramp_pts)
    rise_ramp = np.linspace(base_a, plateau_a, ramp_pts)

    # [1: ] because the first element of descent will be repeated
    descent_t = total_t - ramp_t if total_t_end_to_end else total_t
    t = np.hstack([rise_ramp_times, rise_ramp_times[:-1] + descent_t])
    a = np.hstack([rise_ramp, np.flip(rise_ramp)[1:]])
    return t, a

def grad_echo():
    ## All times are relative to a single TR, starting at time 0
    lo_freq = 2 # MHz
    rf_amp = 0.5 # 1 = full-scale
    trs = 21 # TR repetitions
    
    slice_amp = 0.4
    phase_amps = np.linspace(-0.5, 0.5, trs)
    readout_amp = 0.8
    
    rf_tstart = 100 # us
    rf_tend = 120 # us

    trap_ramp_duration = 30
    trap_ramp_pts = 5 # deliberately chosen to be uneven for testing
    slice_tstart = rf_tstart - trap_ramp_duration
    slice_duration = (rf_tend - rf_tstart) + 2*trap_ramp_duration # includes rise, plateau and fall
    phase_tstart = rf_tend + 60
    phase_duration = 100
    readout_tstart = phase_tstart
    readout_duration = phase_duration*2
    
    rx_tstart = readout_tstart + trap_ramp_duration # us
    rx_tend = readout_tstart + readout_duration - trap_ramp_duration # us
    rx_period = 10/3 # us, 300 kHz rate
    
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
    
    tr_total_time = 400 # start-finish TR time

    def grad_echo_tr(tstart, phase_amp):
        gvxt, gvxa = trapezoid(slice_amp, slice_duration, trap_ramp_duration, trap_ramp_pts)
        gvyt, gvya = trapezoid(phase_amp, phase_duration, trap_ramp_duration, trap_ramp_pts)
        
        gvzt1 = trapezoid(readout_amp, readout_duration/2, trap_ramp_duration, trap_ramp_pts)
        gvzt2 = trapezoid(-readout_amp, readout_duration/2, trap_ramp_duration, trap_ramp_pts)
        gvzt = np.hstack([gvzt1[0], gvzt2[0] + readout_duration/2])
        gvza = np.hstack([gvzt1[1], gvzt2[1]])
        
        value_dict = {
            'tx0': ( np.array([rf_tstart, rf_tend]) + tstart, np.array([rf_amp, 0]) ),
            'grad_vx': ( gvxt + tstart + slice_tstart, gvxa ),
            'grad_vy': ( gvyt + tstart + phase_tstart, gvya),
            'grad_vz': ( gvzt + tstart + readout_tstart, gvza),
            'rx0_rst_n': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
            'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period)

    tr_t = 20 # start the first TR at 20us
    for phase_amp in phase_amps:
        expt.add_flodict( grad_echo_tr( tr_t, phase_amp) )
        tr_t += tr_total_time

    rxd, msgs = expt.run()
    expt.close_server()

    rxr = rxd[4]['run_seq']
    if False:
        plt.plot( np.array(rxr['rx0_i'], dtype=np.int32) )
        plt.plot( np.array(rxr['rx0_q'], dtype=np.int32) )
        plt.show()

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
    
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period)

    tr_t = 20 # start the first TR at 20us
    for th in angles:
        expt.add_flodict( radial_tr( tr_t, th ) )
        tr_t += tr_total_time

    rxd, msgs = expt.run()
    expt.close_server()

    rxr = rxd[4]['run_seq']
    if False:
        plt.plot( np.array(rxr['rx0_i'], dtype=np.int32) )
        plt.plot( np.array(rxr['rx0_q'], dtype=np.int32) )
        plt.show()

if __name__ == "__main__":
    from local_config import grad_board
    assert grad_board == "ocra1", "Please run examples with OCRA1; GPA-FHDO tests not yet ready"
    # import cProfile
    # cProfile.run('test_grad_echo_loop()')
    grad_echo()
    # radial()
