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

def grad_echo(trs=21, plot_rx=False, init_gpa=False, 
              dbg_sc=1, # set to 0 to avoid 2nd RF debugging pulse, otherwise amp between 0 or 1
              lo_freq=0.05, # MHz
              rf_amp=1, # 1 = full-scale
              
              slice_amp=0.4, # 1 = gradient full-scale
              phase_amp=0.3, # 1 = gradient full-scale
              readout_amp=0.8, # 1 = gradient full-scale
              rf_duration=100,
              trap_ramp_duration=50, # us, ramp-up/down time
              trap_ramp_pts=20, # how many points to subdivide ramp into
              phase_delay=100, # how long after RF end before starting phase ramp-up
              phase_duration=500, # length of phase plateau
              tr_wait=100, # delay after end of RX before start of next TR

              rx_period = 10/3 # us, 300 kHz RX sampling rate
              ):
    ## All times are in the context of a single TR, starting at time 0

    phase_amps = np.linspace(phase_amp, -phase_amp, trs)
    
    rf_tstart = 100 # us
    rf_tend = rf_tstart + rf_duration # us

    slice_tstart = rf_tstart - trap_ramp_duration
    slice_duration = (rf_tend - rf_tstart) + 2*trap_ramp_duration # includes rise, plateau and fall
    phase_tstart = rf_tend + phase_delay
    readout_tstart = phase_tstart
    readout_duration = phase_duration*2
    
    rx_tstart = readout_tstart + trap_ramp_duration # us
    rx_tend = readout_tstart + readout_duration - trap_ramp_duration # us
    
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
    
    tr_total_time = readout_tstart + readout_duration + tr_wait # start-finish TR time

    def grad_echo_tr(tstart, pamp):
        gvxt, gvxa = trapezoid(slice_amp, slice_duration, trap_ramp_duration, trap_ramp_pts)
        gvyt, gvya = trapezoid(pamp, phase_duration, trap_ramp_duration, trap_ramp_pts)
        
        gvzt1 = trapezoid(readout_amp, readout_duration/2, trap_ramp_duration, trap_ramp_pts)
        gvzt2 = trapezoid(-readout_amp, readout_duration/2, trap_ramp_duration, trap_ramp_pts)
        gvzt = np.hstack([gvzt1[0], gvzt2[0] + readout_duration/2])
        gvza = np.hstack([gvzt1[1], gvzt2[1]])

        rx_tcentre = (rx_tstart + rx_tend) / 2
        value_dict = {
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,   rx_tcentre - 10, rx_tcentre + 10]) + tstart,
                     np.array([rf_amp,0,  dbg_sc*(1 + 0.5j),0]) ),
            'grad_vx': ( gvxt + tstart + slice_tstart, gvxa ),
            'grad_vy': ( gvyt + tstart + phase_tstart, gvya),
            'grad_vz': ( gvzt + tstart + readout_tstart, gvza),
            'rx0_rst_n': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
            'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa)

    tr_t = 20 # start the first TR at 20us
    for pamp in phase_amps:
        expt.add_flodict( grad_echo_tr( tr_t, pamp) )
        tr_t += tr_total_time

    rxd, msgs = expt.run()
    expt.close_server(True)

    rxr = rxd[4]['run_seq']
    if plot_rx:
        plt.plot( np.array(rxr['rx0_i'], dtype=np.int32) )
        plt.plot( np.array(rxr['rx0_q'], dtype=np.int32) )
        plt.show()

def radial(trs=36, plot_rx=False, init_gpa=False):
    ## All times are relative to a single TR, starting at time 0
    lo_freq = 0.2 # MHz
    rf_amp = 0.5 # 1 = full-scale
    
    G = 0.5 # Gx = G cos (t), Gy = G sin (t)
    angles = np.linspace(0, 2*np.pi, trs) # angle
    
    grad_tstart = 0 # us
    rf_tstart = 5 # us
    rf_tend = 50 # us
    rx_tstart = 70 # us
    rx_tend = 180 # us
    rx_period = 3 # us
    tr_total_time = 220 # start-finish TR time

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
            'grad_vy': ( np.array([grad_tstart]),
                         np.array([gx]) ),
            'grad_vz': ( np.array([grad_tstart]),
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
    
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa)

    tr_t = 20 # start the first TR at 20us
    for th in angles:
        expt.add_flodict( radial_tr( tr_t, th ) )
        tr_t += tr_total_time

    rxd, msgs = expt.run()
    # expt.close_server(True)

    rxr = rxd[4]['run_seq']
    if plot_rx:
        plt.plot( np.array(rxr['rx0_i'], dtype=np.int32) )
        plt.plot( np.array(rxr['rx0_q'], dtype=np.int32) )
        plt.show()

if __name__ == "__main__":
    from local_config import grad_board
    assert grad_board == "ocra1", "Please run examples with OCRA1; GPA-FHDO tests not yet ready"
    # import cProfile
    # cProfile.run('test_grad_echo_loop()')
    grad_echo(trs=10, init_gpa=True)
    # radial(trs=100, init_gpa=True)
