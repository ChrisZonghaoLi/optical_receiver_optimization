import numpy as np
import pandas as pd
from ckt import *
from utils import *
from eq import *
from utils import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from statistical_eye import statistical_eye

import os
import sys

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
plt.style.use(style='default')
plt.rcParams['font.family']='calibri'


class Channel(object):
    
    def __init__(self, 
                 M,
                 f,
                 tline_file,
                 data_rate=64e9, # baud rate
                 samples_per_symbol=128, 
                 num_symbols=1000,
                 target_BER = 2.4e-4,
                 current_amplitude=100e-6,
                 rise_fall_time = 1/64e9*0.4,
                 beta = 0.35,
                 z0=50):
        
        self.data_rate = data_rate
        self.samples_per_symbol = samples_per_symbol
        self.target_BER = target_BER
        self.M = M
        self.f = f
        self.s = 2*np.pi*f*1j
        self.z0 = z0
        self.tline_file = tline_file
        self.num_symbols = num_symbols
        self.current_amplitude = current_amplitude
        self.rise_fall_time = rise_fall_time
        self.t_symbol = 1/data_rate
        self.t_sample = self.t_symbol/samples_per_symbol # sample time duration
        
        self.channel_abcd = np.zeros((2,2,len(f)), dtype=complex)
        self.num_samples = samples_per_symbol * num_symbols
        self.rise_and_fall_sample_number = int(np.floor(rise_fall_time/self.t_sample))
        
        pulse_input = np.zeros(self.num_samples)
        for i1 in range(1, self.rise_and_fall_sample_number+1):
            pulse_input[int(self.num_samples/2)+i1] = i1/self.rise_and_fall_sample_number
        for i2 in range(1, samples_per_symbol-1*self.rise_and_fall_sample_number+1):
            pulse_input[int(self.num_samples/2)+i1+i2] = 1
        for i3 in range(1, self.rise_and_fall_sample_number+1):
            pulse_input[int(self.num_samples/2)+i1+i2+i3] = 1-i3/self.rise_and_fall_sample_number
                
        self.pulse_input = current_amplitude * pulse_input    
        
    def pulse_filter(self, time_constant):
        """
        This is a STC filter for the PD pulse input

        Parameters
        ----------
        time_constant : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        w0 = 1/time_constant
        self.H_pulse_filter = 1/(1+self.s/w0)
        return self.H_pulse_filter

    def noise_filter(self, freq_pole=40e9, plot=False):
        self.H_noise = 1/(1+self.s/(2*np.pi*freq_pole))**2
        # for second order filter, 3dB bandwidth is not equal to pole frequency, therefore:
        freq_3dB = freq_pole/1.554
        # https://analog.intgckts.com/equivalent-noise-bandwidth/, using 1.22 for second order filter
        self.f_enbw = 1.22*freq_3dB
        
        if plot==True:
            plt.plot(self.f,20*np.log10(abs(self.H_noise)),linewidth=2)
            plt.xscale('log')
            plt.xlim(left=1e9)
            plt.xlabel('Frequency (Hz)', weight='bold')
            plt.ylabel('$\mathbf{{|H_{noise}|} (dB)}$', weight='bold')
            plt.title(f'Noise filter with two poles at {freq_pole/1e9}GHz')
            plt.grid()
            plt.show()
            
        return self.H_noise, self.f_enbw

    def noise_laser(self, f_enbw, target_snr_db=20, mean_noise=0):
        #signal_watts = self.pulse_input ** 2
        #signal_avg_watts = np.mean(signal_watts)
        signal_avg_watts = np.max(self.current_amplitude ** 2)
        signal_avg_db = 10*np.log10(signal_avg_watts)
        
        noise_avg_db = signal_avg_db - target_snr_db
        noise_avg_watts = 10**(noise_avg_db/10)
        # https://radarsp.weebly.com/uploads/2/1/4/7/21471216/snr_of_a_simple_pulse_in_noise.pdf
        self.S_laser_noise = noise_avg_watts/self.f_enbw # PSD
        return self.S_laser_noise

    def photo_detector(self, Cpd, Rpd):
        self.pd_abcd = photo_detector(Cpd, Rpd, self.f)
        return self.pd_abcd

    def tia(self, gm, Rf, Ca, ft, Ztot_precede=None):
        tia = TIA(gm, Rf, Ca, ft, self.f)
        self.tia_abcd = tia.abcd()
        self.S_tia_noise =  tia.tia_output_noise_psd(Ztot_precede)
        self.tia_trans_impedance = 1/self.tia_abcd[:,1,0]
        self.sigma_squared_tia = np.trapz(self.S_tia_noise, self.f)
        return self.tia_abcd, self.S_tia_noise, self.tia_trans_impedance, self.sigma_squared_tia

    def tline(self, length, width, dataset='old'):
        self.tline_abcd, _= tline(self.tline_file, length, width, dataset=dataset)
        return self.tline_abcd

    def bump_tx(self, Lseries_bump_tx, Cshunt_bump_tx, mode='tx'):
        self.bump_tx_abcd = bump(Lseries_bump_tx, Cshunt_bump_tx, self.f, mode)
        return self.bump_tx_abcd
        
    def bump_rx(self, Lseries_bump_rx, Cshunt_bump_rx, mode='rx'):
        self.bump_rx_abcd = bump(Lseries_bump_rx, Cshunt_bump_rx, self.f, mode)
        return self.bump_rx_abcd
        
    def pad(self, Cpad):
        self.pad_abcd = admittance2abcd(1j*2*np.pi*self.f*Cpad)
        return self.pad_abcd

    def tcoil(self, L, Cesd, k):
        '''
        This is just a simplified symmetric t-coil math model, not realistic, but you can use it anyway if you want

        '''
        self.tcoil_abcd = tcoil(L, Cesd, k, self.f)
        return self.tcoil_abcd

    def ffe(self, h, tap_weights, n_taps_pre, n_taps_post):
        self.h_ffe = ffe(h, tap_weights, n_taps_pre, n_taps_post, self.samples_per_symbol)
        return self.h_ffe

    def dfe(self, signal_in, tap_weights_dfe):
        self.pulse_response_dfe = dfe(signal_in, tap_weights_dfe, self.samples_per_symbol)
        return self.pulse_response_dfe

    def abcd2H(self, channel_abcd):
        """
        ABCD matrix converted to channel frequency response H, this is assuming that H = V(out)/I(in)
        which is essentially 1/C

        Parameters
        ----------
        channel_abcd : complex array
            Channel ABCD matrix

        Returns
        -------
        self.H: complex array
            channel frequency response H.

        """
        self.channel_abcd = channel_abcd
        self.H = 1/self.channel_abcd[:,1,0]
        return self.H

    def H2h(self, H):
        """
        frequency response H converted to impulse response h

        Parameters
        ----------
        H : complex array
            channel frequency response.

        Returns
        -------
        self.h: array
            channel impulse response

        """
        self.h, self.t, _ = freq2impulse(H, self.f)
        return self.h, self.t
    
    def h2pulse(self, h):
        """
        impulse response h to pulse response

        Parameters
        ----------
        h : array
            channel impulse response h.

        Returns
        -------
        self.pulse_response: array
            channel pulse response

        """
        self.pulse_response = np.convolve(h, self.pulse_input)
        return self.pulse_response
    
    def channel_coefficients(self, pulse_response, main_idx, n_precursors, n_postcursors, *args, **kwargs):
        self.pulse_response_coefficients = channel_coefficients(pulse_response, main_idx, self.t_sample*np.array(list(range(len(pulse_response)))), self.samples_per_symbol, n_precursors, n_postcursors, *args, **kwargs) 
        return self.pulse_response_coefficients




    
    

