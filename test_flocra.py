#!/usr/bin/env python3
# Basic hacks and tests for flocra system
#

import numpy as np
import matplotlib.pyplot as plt

import socket, time
from local_config import ip_address, port, grad_board, fpga_clk_freq_MHz
from server_comms import *

import pdb
st = pdb.set_trace

IFINISH = 0x1
IWAIT = 0x2
ITRIG = 0x3
ITRIGFOREVER=0x4
IDATA = 0x80

GRAD_CTRL = 0
GRAD_LSB = 1
GRAD_MSB = 2
RX0_CTRL = 3
RX1_CTRL = 4
TX0_I = 5
TX0_Q = 6
TX1_I = 7
TX1_Q = 8
DDS0_PHASE_LSB = 9
DDS0_PHASE_MSB = 10
DDS1_PHASE_LSB = 11
DDS1_PHASE_MSB = 12
DDS2_PHASE_LSB = 13
DDS2_PHASE_MSB = 14
GATES_LEDS = 15

STATE_IDLE = 0
STATE_PREPARE = 1
STATE_RUN = 2
STATE_COUNTDOWN = 3
STATE_TRIG = 4
STATE_TRIG_FOREVER = 5
STATE_HALT = 8

def insta(instr, data):
    """ Instruction A: FSM control """
    assert instr in [IFINISH, IWAIT, ITRIG, ITRIGFOREVER], "Unknown instruction"
    return (instr << 24) | (data & 0xffffff)
    
def instb(tgt, delay, data):
    """ Instruction B: timed buffered data """
    assert tgt <= 24, "Unknown target buffer"
    assert 0 <= delay <= 255, "Delay out of range"
    assert (data & 0xffff) == (data & 0xffffffff), "Data out of range"
    return (IDATA << 24) | ( (tgt & 0x7f) << 24 ) | ( (delay & 0xff) << 16 ) | (data & 0xffff)

def get_exec_state(socket, display=True):
    reply, status = command({'regstatus': 0}, socket, print_infos=True)
    exec_reg = reply[4]['regstatus'][0]
    state = exec_reg >> 24
    pc = exec_reg & 0xffffff
    if display:
        print('state: {:d}, PC: {:d}'.format(state, pc))
    return state, pc

def run_manual_test(data, interval=0.001, timeout=20):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        # mem data
        # data = np.zeros(65536, dtype=np.uint32)
        command( {'flo_mem': data.tobytes()} , s)

        # Start execution
        command({'ctrl': 0x1}, s, print_infos=True)

        for k in range(int(timeout/interval)):
            time.sleep(interval)
            state, pc = get_exec_state(s)
            if (state == STATE_HALT):
                # one final acquisition to kill time on the simulation
                get_exec_state(s)
                break

        # Stop execution
        command({'ctrl': 0x2}, s, print_infos=True)

        # Flush RX data
        rx_data = command ({'flush_rx':0}, s, print_infos=True)[0][4]['flush_rx']

        # Auto-close server if it's simulating
        if command({'are_you_real':0}, s)[0][4]['are_you_real'] == "simulation":
            send_packet(construct_packet({}, 0, command=close_server_pkt), s)

        return rx_data
    
def run_streaming_test(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        if False: # clear memory first
            command( {'flo_mem': np.zeros(65536, dtype=np.uint32).tobytes()} , s)        

        # mem data
        rx_data, msgs = command({'run_seq': data.tobytes()} , s)

        print(msgs)

        # Auto-close server if it's simulating
        if command({'are_you_real':0}, s)[0][4]['are_you_real'] == "simulation":
            send_packet(construct_packet({}, 0, command=close_server_pkt), s)

        return rx_data    

def leds():
    time_interval_s = 4 # how long the total sequence should run for
    total_states = 256
    delay = int(np.round(time_interval_s * fpga_clk_freq_MHz * 1e6 / total_states))
    
    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0
    for k in range(256):
        raw_data[addr] = insta(IWAIT, delay - 1); addr += 1 # -1 because the next instruction takes 1 cycle
        # raw_data[addr] = insta(IWAIT, 0x300000); addr += 1 # -1 because the next instruction takes 1 cycle
        raw_data[addr] = instb(GATES_LEDS, 0, (k & 0xff) << 8); addr += 1
    # go idle
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def tx_short():
    lo_freq0 = 5 # MHz
    lo_freq1 = 10
    lo_freq2 = 1.5
    lo_freq3 = 13.333333
    lo_amp = 100 # percent

    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0

    # Turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in flocra
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in flocra    
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"    

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, 0); addr += 1    
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, 0x8000); addr += 1    

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 20); addr += 1 # max I value
    
    steps = 300
    pattern0 = np.hstack([ np.linspace(1, 0, steps//3), np.linspace(1, -1, steps//3), np.linspace(-1, 1, steps//3) ])
    pattern1 = np.hstack([ np.linspace(0, 1, steps//3), np.linspace(1, 0, steps//3), np.linspace(0, -1, steps//3) ])    
    tdata0 = np.round(0x7fff * pattern0).astype(np.int16)
    tdata1 = np.round(0x7fff * pattern1).astype(np.int16)    

    for k in range(steps):
        hv0 = tdata0[k] & 0xffff
        hv1 = tdata1[k] & 0xffff
        raw_data[addr] = instb(TX0_I, 3, hv0); addr += 1
        raw_data[addr] = instb(TX0_Q, 2, hv0); addr += 1
        raw_data[addr] = instb(TX1_I, 1, hv1); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, hv1); addr += 1

    # do mid-scale output, and change frequency
    raw_data[addr] = instb(TX0_I, 3, 0x4000); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x4000); addr += 1

    # wait for 3us; shortened delay by 4 cycles for the next instructions to be right on time
    raw_data[addr] = insta(IWAIT, int(3 * fpga_clk_freq_MHz) - 4); addr += 1 

    dds2_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq2).astype(np.uint32) # 31b phase accumulator in flocra
    dds3_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq3).astype(np.uint32) # 31b phase accumulator in flocra    
    
    # switch frequency on both channels simultaneously, no reset
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds2_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds2_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds3_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds3_phase_step >> 16); addr += 1

    raw_data[addr] = insta(IWAIT, int(4 * fpga_clk_freq_MHz)); addr += 1 
    
    # go idle
    raw_data[addr] = instb(TX0_I, 3, 0); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x8000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x8000); addr += 1
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def rx_short():
    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0

    # turn DDS source 0 on
    lo_freq0 = 5 # MHz
    cic_decimation = 4
    dds_demod_ch = 0
    acquisition_ticks = 800
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in flocra
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    raw_data[addr] = instb(DDS0_PHASE_LSB, 1, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, dds0_phase_step >> 16); addr += 1
    
    # configure RX settings: both channels to use DDS source 3 (DC), decimation of 10
    # reset CICs
    raw_data[addr] = instb(RX0_CTRL, 0, 0x0000); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    # take them out of reset later
    raw_data[addr] = instb(RX0_CTRL, 50, 0x8000 | cic_decimation | (dds_demod_ch << 12) ); addr += 1
    # briefly signal that there's a new rate
    raw_data[addr] = instb(RX0_CTRL, 50, 0xc000 | cic_decimation | (dds_demod_ch << 12) ); addr += 1    
    # raw_data[addr] = instb(RX1_CTRL, 49, 0xf000 | cic_decimation); addr += 1
    # end the new rate flag (the buffers are not empty so no offset time is needed)
    raw_data[addr] = instb(RX0_CTRL, 0, 0x8000 | cic_decimation | (dds_demod_ch << 12) ); addr += 1 # decimation may not be needed here
    # raw_data[addr] = instb(RX1_CTRL, 0, 0xb000 | cic_decimation); addr += 1

    # just acquire data for a while
    raw_data[addr] = insta(IWAIT, acquisition_ticks - 1); addr += 1

    # reset RX again
    raw_data[addr] = instb(RX0_CTRL, 1, 0x3000); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x3000); addr += 1

    # turn off DDS0
    raw_data[addr] = instb(DDS0_PHASE_LSB, 1, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, 0); addr += 1

    # go idle
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def loopback():
    lo_freq0 = 4 # MHz
    lo_freq1 = 4
    lo_freq2 = 1.5
    lo_freq3 = 1.5
    lo_amp = 100 # percent

    cic0_decimation = 4
    cic1_decimation = 7
    dds_demod_ch = 1

    extra_time = 20

    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0

    # Turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in flocra
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in flocra    
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"    

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 7, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 6, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 5, 0); addr += 1    
    raw_data[addr] = instb(DDS1_PHASE_MSB, 4, 0x8000); addr += 1    

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 0, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 0, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    # configure RX settings: both channels to use DDS source 3 (DC), decimation of 10
    # reset CICs
    raw_data[addr] = instb(RX0_CTRL, 0, 0x0000); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    # take them out of reset later
    raw_data[addr] = instb(RX0_CTRL, 40, 0x8000 | cic0_decimation | (dds_demod_ch << 12) ); addr += 1
    raw_data[addr] = instb(RX1_CTRL, 39, 0x8000 | cic1_decimation | (dds_demod_ch << 12) ); addr += 1    
    # briefly signal that there's a new rate
    raw_data[addr] = instb(RX0_CTRL, 40, 0xc000 | cic0_decimation | (dds_demod_ch << 12) ); addr += 1
    raw_data[addr] = instb(RX1_CTRL, 40, 0xc000 | cic1_decimation | (dds_demod_ch << 12) ); addr += 1    
    # raw_data[addr] = instb(RX1_CTRL, 49, 0xf000 | cic_decimation); addr += 1
    # end the new rate flag (the buffers are not empty so no offset time is needed)
    raw_data[addr] = instb(RX0_CTRL, 0, 0x8000 | cic0_decimation | (dds_demod_ch << 12) ); addr += 1 # decimation may not be needed here
    raw_data[addr] = instb(RX1_CTRL, 0, 0x8000 | cic1_decimation | (dds_demod_ch << 12) ); addr += 1    
    # raw_data[addr] = instb(RX1_CTRL, 0, 0xb000 | cic_decimation); addr += 1    
    
    steps = 300
    pattern0 = np.hstack([ np.linspace(1, 0, steps//3), np.linspace(1, -1, steps//3), np.linspace(-1, 1, steps//3) ])
    # pattern1 = np.hstack([ np.linspace(0, 1, steps//3), np.linspace(1, 0, steps//3), np.linspace(0, -1, steps//3) ])
    pattern1 = pattern0
    tdata0 = np.round(0x7fff * pattern0).astype(np.int16)
    tdata1 = np.round(0x7fff * pattern1).astype(np.int16)    

    for k in range(steps):
        hv0 = tdata0[k] & 0xffff
        hv1 = tdata1[k] & 0xffff
        raw_data[addr] = instb(TX0_I, 3, hv0); addr += 1
        raw_data[addr] = instb(TX0_Q, 2, hv0); addr += 1
        raw_data[addr] = instb(TX1_I, 1, hv1); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, hv1); addr += 1

    # do mid-scale output, and change frequency
    raw_data[addr] = instb(TX0_I, 3, 0x4000); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x4000); addr += 1

    # wait for 3us; shortened delay by 4 cycles for the next instructions to be right on time
    raw_data[addr] = insta(IWAIT, int(3 * fpga_clk_freq_MHz) - 4); addr += 1 

    dds2_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq2).astype(np.uint32) # 31b phase accumulator in flocra
    dds3_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq3).astype(np.uint32) # 31b phase accumulator in flocra    
    
    # switch frequency on both channels simultaneously, no reset
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds2_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds2_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds3_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds3_phase_step >> 16); addr += 1

    # wait for extra acquisition time
    raw_data[addr] = insta(IWAIT, int(extra_time * fpga_clk_freq_MHz)); addr += 1

    # end RX
    raw_data[addr] = instb(RX0_CTRL, 1, 0x0000); addr += 1
    raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    
    # go idle
    raw_data[addr] = instb(TX0_I, 3, 0); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x8000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x8000); addr += 1
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def long_loopback():
    lo_freq0 = 4 # MHz
    lo_freq1 = 4
    lo_freq2 = 1.5
    lo_freq3 = 1.5
    lo_amp = 100 # percent

    # These settings lead to memory-buffer-low events
    if True:
        cic0_decimation = 50
        cic1_decimation = 94
        extra_time = 20
        sine_ts = 5
        max_addr = 141000

    # These settings lead to a very long sequence, with a few memory buffer low warnings
    if False:
        cic0_decimation = 100
        cic1_decimation = 200
        extra_time = 20
        sine_ts = 10
        max_addr = 500000
    
    dds_demod_ch = 3
    raw_data = np.zeros(max_addr, dtype=np.uint32) # massive sequence
    addr = 0

    # Turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in flocra
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in flocra    
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"    

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 7, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 6, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 5, 0); addr += 1    
    raw_data[addr] = instb(DDS1_PHASE_MSB, 4, 0x8000); addr += 1

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 0, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 0, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    # configure RX settings: both channels to use DDS source 3 (DC), decimation of 10
    # reset CICs
    raw_data[addr] = instb(RX0_CTRL, 0, 0x0000); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    # take them out of reset later
    raw_data[addr] = instb(RX0_CTRL, 40, 0x8000 | cic0_decimation | (dds_demod_ch << 12) ); addr += 1
    raw_data[addr] = instb(RX1_CTRL, 39, 0x8000 | cic1_decimation | (dds_demod_ch << 12) ); addr += 1    
    # briefly signal that there's a new rate
    raw_data[addr] = instb(RX0_CTRL, 40, 0xc000 | cic0_decimation | (dds_demod_ch << 12) ); addr += 1
    raw_data[addr] = instb(RX1_CTRL, 40, 0xc000 | cic1_decimation | (dds_demod_ch << 12) ); addr += 1    
    # raw_data[addr] = instb(RX1_CTRL, 49, 0xf000 | cic_decimation); addr += 1
    # end the new rate flag (the buffers are not empty so no offset time is needed)
    raw_data[addr] = instb(RX0_CTRL, 0, 0x8000 | cic0_decimation | (dds_demod_ch << 12) ); addr += 1 # decimation may not be needed here
    raw_data[addr] = instb(RX1_CTRL, 0, 0x8000 | cic1_decimation | (dds_demod_ch << 12) ); addr += 1    
    # raw_data[addr] = instb(RX1_CTRL, 0, 0xb000 | cic_decimation); addr += 1    
    
    steps = 300
    pattern0 = np.hstack([ np.linspace(1, 0, steps//3), np.linspace(1, -1, steps//3), np.linspace(-1, 1, steps//3) ])
    # pattern1 = np.hstack([ np.linspace(0, 1, steps//3), np.linspace(1, 0, steps//3), np.linspace(0, -1, steps//3) ])
    pattern1 = pattern0
    tdata0 = np.round(0x7fff * pattern0).astype(np.int16)
    tdata1 = np.round(0x7fff * pattern1).astype(np.int16)    

    for k in range(steps):
        hv0 = tdata0[k] & 0xffff
        hv1 = tdata1[k] & 0xffff
        raw_data[addr] = instb(TX0_I, 3, hv0); addr += 1
        raw_data[addr] = instb(TX0_Q, 2, hv0); addr += 1
        raw_data[addr] = instb(TX1_I, 1, hv1); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, hv1); addr += 1

    # do mid-scale output, and change frequency
    raw_data[addr] = instb(TX0_I, 3, 0x4000); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x4000); addr += 1

    # wait for 3us; shortened delay by 4 cycles for the next instructions to be right on time
    raw_data[addr] = insta(IWAIT, int(3 * fpga_clk_freq_MHz) - 4); addr += 1 

    dds2_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq2).astype(np.uint32) # 31b phase accumulator in flocra
    dds3_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq3).astype(np.uint32) # 31b phase accumulator in flocra    
    
    # switch frequency on both channels simultaneously, no reset
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds2_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds2_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds3_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds3_phase_step >> 16); addr += 1

    # wait for extra acquisition time
    raw_data[addr] = insta(IWAIT, int(extra_time * fpga_clk_freq_MHz)); addr += 1

    ### Long sinusoid of slowly increasing frequency
    start_addr = addr
    end_addr = max_addr - 1000 # reserve 1000 spots at the end for final commands
    phase_points = (end_addr - start_addr) // 5
    t = np.linspace(0, 1, phase_points)
    c = np.cos(2*np.pi*20*t) * (0.2 + 0.5*t)
    s = np.sin(2*np.pi*13*t) * (0.1 + 0.8*t)
    cb = np.round(c * 0x7fff).astype(np.int16) & 0xffff
    sb = np.round(s * 0x7fff).astype(np.int16) & 0xffff
    
    for k in range(phase_points):
        raw_data[addr] = instb(TX0_I, 0, cb[k]); addr += 1
        raw_data[addr] = instb(TX0_Q, 0, sb[k]); addr += 1
        raw_data[addr] = instb(TX1_I, 0, cb[k]); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, sb[k]); addr += 1
        raw_data[addr] = insta(IWAIT, sine_ts); addr += 1
    
    # end RX
    raw_data[addr] = instb(RX0_CTRL, 1, 0x0000); addr += 1
    raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    
    # go idle
    raw_data[addr] = instb(TX0_I, 0, 0); addr += 1
    raw_data[addr] = instb(TX0_Q, 0, 0x8000); addr += 1
    raw_data[addr] = instb(TX1_I, 0, 0); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x8000); addr += 1
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    print(addr)
    return raw_data

if __name__ == "__main__":
    # for k in range(1000):
    # run_test(rx_short(), interval=0.01, timeout=20)

    if False:
        data = long_loopback()
        st()
    
    if True:
        # clear mem
        
        res = run_streaming_test(long_loopback())

        rxd = res[4]['run_seq']
        rx0_i = np.array(rxd['rx0_i'], dtype=np.int32)
        rx0_q = np.array(rxd['rx0_q'], dtype=np.int32)+1e8
        rx1_i = np.array(rxd['rx1_i'], dtype=np.int32)+2e8
        rx1_q = np.array(rxd['rx1_q'], dtype=np.int32)+3e8    
        plt.plot(rx0_i)
        plt.plot(rx0_q)
        plt.plot(rx1_i)
        plt.plot(rx1_q)

        plt.legend(['rx0_i', 'rx0_q', 'rx1_i', 'rx1_q'])

        plt.show()
    
    if False:
        trials = 1
        for k in range(trials):
            rxd = run_manual_test(loopback(), interval=0.01, timeout=20)
            rx0 = np.array(rxd['ch0'], dtype=np.int32)
            rx1 = np.array(rxd['ch1'], dtype=np.int32)        
            plt.plot(rx0[::2])
            plt.plot(rx0[1::2])
            plt.plot(rx1[::2])
            plt.plot(rx1[1::2])        

        plt.xlabel('sample')
        plt.title('loopback, {:d} trials'.format(trials))
        plt.show()
