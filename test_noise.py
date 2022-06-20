#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import experiment as ex
from local_config import grad_board

import pdb
st = pdb.set_trace

def test_rf_jitter(loops=10):
    """Two successive pulses, with an adjustable interval between them.

    Trigger an oscilloscope from the first pulse, and look at the
    second with a suitable hold-off time. Over many repetitions, the
    timing jitter will be visible.
    """
    interval = 5 # us
    pulse_length = 2 # us
    pulse_freq = 2 # MHz
    rf_amp = 0.8
    exp = ex.Experiment(lo_freq=pulse_freq)

    rf_data = (
        np.array([10, 10 + pulse_length, 10 + interval, 10 + pulse_length + interval]),
        np.array([rf_amp, 0, rf_amp, 0])
    )

    event_dict = {'tx0': rf_data, 'tx1': rf_data}
    exp.add_flodict(event_dict)

    for k in range(loops):
        exp.run()
        print(k)

    exp.close_server(only_if_sim=True)

if __name__ == "__main__":
    test_rf_jitter(loops=int(1e6))
