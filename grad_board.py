#!/usr/bin/env python3
# 
# Classes to handle GPA initialisation, calibration and communication
#
# They need to at least implement the following methods:
#
# init_hw() to program the GPA chips on power-up or reset them if
# they're in an undefined state, as well as configure the FPGA with a
# suitable gradient BRAM and SPI timing divisors
#
# write_dac() to send binary numbers directly to a DAC (the method
# should take care of bit shifts, extra bits etc - the user supplies
# only the binary DAC output code)
#
# read_adc() to retrieve a binary ADC word from the GPA (if enabled);
# should output the binary ADC code, but shouldn't be responsible for
# off-by-one codes (i.e. for the GPA-FHDO, it doesn't have to correct
# the ADC's behaviour of sending the previously-read voltage with the
# current transfer)
#
# calibrate() to prepare the data for user-defined calibration
# procedures (such as scaling/offset, piecewise interpolation, etc) to
# take place later. This might be outputting test currents and
# measuring the actual current using an ADC, loading a file containing
# manually acquired code-vs-voltage calibration data, etc.
# Once the method has been run, the system should be ready to handle:
#
# float2bin() to convert a list of input Numpy arrays in units of the
# full-scale DAC output (i.e. [-1, 1]) into the binary BRAM data to
# reproduce the multi-channel waveform on the GPA - should apply any
# desired calibrations/transforms internally.
#
# TODO: actually use class inheritance here, instead of two separate classes

import numpy as np

class OCRA1:
    def __init__(self,
                 grad_t,
                 grad_channels,
                 server_command_f,
                 spi_freq=None):

        grad_clk_t = 0.007 # 7ns period
        self.true_grad_div = int(np.round(grad_t/grad_clk_t)) # true divider value
        self.grad_div = self.true_grad_div - 4 # what's sent to server
        self.grad_t = grad_clk_t * self.true_grad_div # gradient DAC update period
        self.grad_channels = grad_channels
        assert 0 < grad_channels < 5, "Strange number of grad channels"

        spi_cycles_per_tx = 30 # actually 24, but including some overhead
        if spi_freq is not None:
            self.spi_div = int(np.floor(1 / (spi_freq*grad_clk_t))) - 1
        else:
            # Auto-decide the SPI freq, to be as low as will work

            # SPI runs in parallel for each channel
            self.true_spi_div = (self.true_grad_div * grad_channels) // spi_cycles_per_tx
            
            if self.true_spi_div > 64:
                self.true_spi_div = 64 # slowest SPI clock possible

        self.spi_div = self.true_spi_div - 1
        self.grad_ser = 0x1 # select which board serialiser is activated on the firmware

        # bind function from Experiment class, or replace with something else for debugging
        self.server_command = server_command_f 

        # Default calibration settings for all channels: linear transformation for now
        self.cal_values = [ (1,0), (1,0), (1,0), (1,0) ]

    def init_hw(self):
        init_words = [
            0x00400004, 0x02400004, 0x04400004, 0x07400004, # reset DACs to power-on values
            0x00200002, 0x02200002, 0x04200002, 0x07200002, # set internal amplifier
            0x00100000, 0x02100000, 0x04100000, 0x07100000, # set outputs to 0
        ]

        # configure grad ctrl divisors
        self.server_command({'grad_div': (self.grad_div, self.spi_div), 'grad_ser': self.grad_ser})

        for iw in init_words:
            # direct commands to grad board
            self.server_command({'grad_dir': iw})

    def write_dac(self, channel, value):
        assert 0, "Not yet written, sorry!"

    def read_adc(self, channel, value):
        assert 0, "OCRA1 has no ADC!"

    def calibrate(self):
        # Fill more in here
        pass

    def float2bin(self, grad_data):
        grad_bram_data = np.zeros(grad_data[0].size * self.grad_channels, dtype=np.uint32)

        for ch, gd in enumerate(grad_data):
            cv = self.cal_values[ch]
            gd_cal = gd * cv[0] + cv[1]
            
            # 2's complement
            gr_dacbits = np.round(131071 * gd_cal).astype(np.uint32) & 0x3ffff 
            gr = (gr_dacbits << 2) | 0x00100000

            # always broadcast for the final channel
            broadcast = ch == self.grad_channels - 1                

            grad_bram_data[ch::self.grad_channels] = gr | (ch << 25) | (broadcast << 24) # interleave data

        return grad_bram_data            

class GPAFHDO:
    def __init__(self,
                 grad_t=2.5,
                 grad_channels=4,
                 spi_freq=None):

        grad_clk_t = 0.007 # 7ns period
        self.true_grad_div = int(np.round(grad_t/grad_clk_t)) # true divider value
        self.grad_div = self.true_grad_div - 4 # what's sent to server
        self.grad_t = grad_clk_t * self.true_grad_div # gradient DAC update period
        self.grad_channels = grad_channels
        assert 0 < grad_channels < 5, "Strange number of grad channels"

        self.gpa_current_per_volt = 3.75 # default value, will be updated by calibrate_gpa_fhdo
        # initialize gpa fhdo calibration with ideal values
        self.dac_values = np.array([0x7000, 0x8000, 0x9000])
        self.gpaCalValues = np.ones((self.grad_channels,self.dac_values.size))

        self.grad_board = local_grad_board
        spi_cycles_per_tx = 30 # actually 24, but including some overhead
        if spi_freq is not None:
            self.spi_div = int(np.floor(1 / (spi_freq*grad_clk_t))) - 1
        else:
            # SPI must be written sequentially for each channel
            self.true_spi_div = self.true_grad_div // spi_cycles_per_tx

            if self.true_spi_div > 64:
                self.true_spi_div = 64 # slowest SPI clock possible

        self.spi_div = self.true_spi_div - 1
        self.grad_ser = 0x2 # select which board serialiser is activated on the firmware

        if self.spi_div < 6:
            warnings.warn('the fastest possible spi_div for GPA FHDO is 6 - check your settings!')

        self.s = socket

    def init_hw(self):
        init_words = [
            0x00030100, # DAC sync reg
            0x40850000, # ADC reset
            0x400b0600, 0x400d0600, 0x400f0600, 0x40110600, # input ranges for each ADC channel
            # TODO: set outputs to ~0
        ]

        # configure grad ctrl divisors
        self.server_command({'grad_div': (self.grad_div, self.spi_div), 'grad_ser': self.grad_ser})        

        for iw in init_words:
            # direct commands to grad board
            self.server_command({'grad_dir': iw})

    def write_dac(self, channel, value):
        self.server_command({'grad_dir': 0x00080000 | (channel<<16) | int(value)}) # DAC output
        
    def read_adc(self, channel):
        self.server_command({'grad_dir': 0x40c00000 | (channel<<18)}) # ADC data transfer
        r, s = self.server_command({'grad_adc': 1}) # ADC data transfer
        return r[4]['grad_adc']

    def expected_adc_code_from_dac_code(self, dac_code):
        """
        a helper function for calibrate_gpa_fhdo(). It calculates the expected adc value for a given dac value if every component was ideal.
        The dac codes that pulseq generates should be based on this ideal assumption. Imperfections will be automatically corrected by calibration.
        """
        dac_voltage = 5 * dac_code / 0xffff
        v_ref = 2.5
        gpa_current = (dac_voltage-v_ref) * self.gpa_current_per_volt
        r_shunt = 0.2
        adc_voltage = gpa_current*r_shunt+v_ref
        adc_gain = 4.096*1.25   # ADC range register setting has to match this
        adc_code = (adc_voltage/adc_gain * 0xffff/2)
        #print('DAC code {:d}, DAC voltage {:f}, GPA current {:f}, ADC voltage {:f}, ADC code {:d}'.format(dac_code,dac_voltage,gpa_current,adc_voltage,adc_code))
        return adc_code

    def calculate_corrected_dac_code(self,channel,dac_code):
        """
        calculates the correction factor for a given dac code by doing linear interpolation on the data points collected during calibration
        """
        return np.interp(self.expected_adc_code_from_dac_code(dac_code),self.gpaCalValues[channel],self.dac_values).astype(np.uint32)

    def ampere_to_dac_code(self, ampere):
        v_ref = 2.5
        dac_code = int( (ampere / self.gpa_current_per_volt + v_ref)/5 * 0xffff)
        return dac_code    

    def calibrate(self,
                  max_current = 2,
                  num_calibration_points = 10,
                  gpa_current_per_volt = 3.75,
                  averages=4,
                  settle_time=0.001, # ms after each write
                  plot=False):
        """
        performs a calibration of the gpa fhdo for every channel. The number of interpolation points in self.dac_values can
        be adapted to the accuracy needed.
        """
        self.gpa_current_per_volt = gpa_current_per_volt
        self.dac_values = np.round(np.linspace(self.ampere_to_dac_code(-max_current),self.ampere_to_dac_code(max_current),num_calibration_points))
        self.dac_values = self.dac_values.astype(int)
        self.gpaCalValues = np.ones((self.grad_channels,self.dac_values.size))
        for channel in range(self.grad_channels):
            if False:
                np.random.shuffle(self.dac_values) # to ensure randomised acquisition
            adc_values = np.zeros([self.dac_values.size, averages]).astype(np.uint32)
            gpaCalRatios = np.zeros(self.dac_values.size)
            for k, dv in enumerate(self.dac_values):
                self.write_gpa_dac(channel,dv)
                sleep(settle_time) # wait 1ms to settle
                
                self.read_gpa_adc(channel) # dummy read
                for m in range(averages): 
                    adc_values[k][m] = self.read_gpa_adc(channel)
                self.gpaCalValues[channel][k] = adc_values.sum(1)[k]/averages
                gpaCalRatios[k] = self.gpaCalValues[channel][k]/self.expected_adc_code_from_dac_code(dv)
                #print('Received ADC code {:d} -> expected ADC code {:d}'.format(int(adc_values.sum(1)[k]/averages),self.expected_adc_code(dv)))
            self.write_gpa_dac(channel,0x8000) # set gradient current back to 0

            if np.amax(gpaCalRatios) > 1.01 or np.amin(gpaCalRatios) < 0.99:
                print('Calibration for channel {:d} seems to be incorrect. Make sure a gradient coil is connected and gpa_current_per_volt value is correct.'.format(channel))
            if plot:
                plt.plot(self.dac_values, adc_values.min(1), 'y.')
                plt.plot(self.dac_values, adc_values.max(1), 'y.')
                plt.plot(self.dac_values, adc_values.sum(1)/averages, 'b.')
                plt.xlabel('DAC word'); plt.ylabel('ADC word, {:d} averages'.format(averages))
                plt.grid(True)
                plt.show()    

    def float2bin(self, grad_data):
        grad_bram_data = np.zeros(grad_data[0].size * self.grad_channels, dtype=np.uint32)

        for ch, gd in enumerate(grad_data):
            # Not 2's complement - 0x0 word is ~0V (-10A), 0xffff is ~+5V (+10A)
            gr_dacbits = np.round(32767.49 * (gd + 1)).astype(np.uint32) & 0xffff
            gr_dacbits_cal = self.calculate_corrected_dac_code(ch,gr_dacbits)                            
            gr = gr_dacbits_cal | 0x80000 | (ch << 16)
            
            # always broadcast for the final channel (TODO: probably not needed for GPA-FHDO, check then remove)
            broadcast = ch == self.grad_channels - 1                

            grad_bram_data[ch::self.grad_channels] = gr | (ch << 25) | (broadcast << 24) # interleave data

        return grad_bram_data
