#!/usr/bin/env python3
#
# Calibration toolbox for the gradients, TX etc, for use with the experiment API/class

# VN: spun calibration out into a separate class.
#
# Need to calibrate data BEFORE its floating-point to DAC-word
# conversion; otherwise loss of precision occurs, because correction
# is done after rounding is performed -- should be done earlier, when
# floating-point data is first introduced into the Experiment
# class. Also, if compile_grad_data() is called multiple times but the
# full gradient data is only partially extended at each call and not
# cleared, this would be performed multiple times unnecessarily.

class CalibratorGpaFhdo:
    def __init__(self, sdfsdsfd):
        # set default calibration coefficients
        self.cal_values = [ (1,0), (1,0), (1,0), (1,0) ]

    def apply_calibration(self, values, channel):
        # TODO: write floating-point code
        gain, offs = self.cal_values[channel] # list of tuples like [ (1,0), (1.1,-0.1) etc ]
        return values * gain + offs

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

    # VN: continue redoing calibration: a) floating-point, b) linear transformation only, c) standalone set of calibration classes used by the experiment class
    
    def calculate_corrected_dac_code(self,channel,dac_code):
        """
        calculates the correction factor for a given dac code by doing linear interpolation on the data points collected during calibration
        """
        return np.interp(self.expected_adc_code_from_dac_code(dac_code),self.gpaCalValues[channel],self.dac_values).astype(np.uint32)

    def ampere_to_dac_code(self, ampere):
        v_ref = 2.5
        dac_code = int( (ampere / self.gpa_current_per_volt + v_ref)/5 * 0xffff)
        return dac_code
    
    def run_calibration(self,
                        max_current = 2,
                        num_calibration_points = 10,
                        gpa_current_per_volt = 3.75,
                        averages=4,
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
                sleep(0.001) # wait 1ms to settle
                
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
