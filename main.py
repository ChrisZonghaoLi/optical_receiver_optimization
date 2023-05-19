'''
    This testbench is running optical link AFE optimization 
    using GA algortithm, it include noises and use custom FoM function.
    We exclude T-coil due to the NDA of GF22-FDX

'''

import numpy as np
import os
import pandas as pd
from utils import *
from eq import *

import skrf as rf

from statistical_eye import statistical_eye

import sys

import multiprocessing

import matplotlib.pyplot as plt
plt.style.use(style='default')
plt.rcParams['font.family']='calibri'

from channel_model import Channel
import time

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m_%d_%Y_%H_%M_%S", named_tuple)

print(time_string)
slicer = 251

tline_dir = './tline/'
file_name = '250u256G.csv'
tline_file = pd.read_csv(tline_dir+file_name).to_numpy()[:slicer,:]
f = tline_file[:,0]
if f[0] == 0: # avoid divided by 0
    f[0] = 1
    
s = 2*np.pi*f*1j

data_rate=64e9
samples_per_symbol = 8
num_symbols=1000
M = 4
sigma_jitter = 0.015
rise_fall_time = 1/data_rate*0.4

channel_model = Channel(M=M,
                    f=f,
                    tline_file = tline_file,
                    data_rate=data_rate, 
                    samples_per_symbol=samples_per_symbol, 
                    num_symbols=num_symbols,
                    beta = 0.35,
                    target_BER = 2.4e-4,
                    current_amplitude=100e-6,
                    rise_fall_time = rise_fall_time,
                    z0=50)


def channel_wrapper(channel_model, action, zf=False):
    
    Cpd=10.7e-15
    Rpd = 87
    Cshunt_bump_rx=10e-15 
    Cpad=100e-15
    freq_step = f[1] - f[0]
    
    tline_length = 1e-3 # action[0] * 0.5e-3 
    tline_width = action[0] * 5e-6 # um
    gm = action[1] * 1e-3 # mS
    Rfb = action[2]
    
    #%% channel_model blocks
    pd_abcd = channel_model.photo_detector(Cpd=Cpd, Rpd=Rpd)
    pd_dbca = abcd2dbca(pd_abcd)
    
    bump_tx_abcd = channel_model.bump_tx(Lseries_bump_tx=20e-12, Cshunt_bump_tx=10e-15)
    bump_tx_dbca= abcd2dbca(bump_tx_abcd)

    # TLine_width = 75e-6 # This can be between 15um and 100um with a step of 5um (i.e. 15um, 20um, 25 um .... 100um).
    # TLine_length = 1e-3 # This can be between 1mm and 20m with a step of 0.5mm (i.e. 1mm, 1.5mm, 2mm .... 20mm).
    tline_abcd = channel_model.tline(length=tline_length, width=tline_width, dataset=dataset)
    tline_dbca= abcd2dbca(tline_abcd)

    bump_rx_abcd = channel_model.bump_rx(Lseries_bump_rx=20e-12, Cshunt_bump_rx=Cshunt_bump_rx)
    bump_rx_dbca= abcd2dbca(bump_rx_abcd)

    pad_abcd = channel_model.pad(Cpad=Cpad)
    pad_dbca = abcd2dbca(pad_abcd)

    # We did not include t-coil here due to the NDA
    '''
    L, W, Nin, Nout = action[3:]
    tcoil_geometry = np.array([L, W, Nin, Nout])
    tcoil_esd_s2p, tcoil_s3p, tcoil_geometry = TcoilEsd(Cesd=80e-15).step(tcoil_geometry)
    tcoil_esd_s2p = tcoil_esd_s2p[:slicer]
    tcoil_s3p = tcoil_s3p[:slicer]
    tcoil_esd_abcd = s2abcd(tcoil_esd_s2p, f)
    tcoil_esd_dbca = abcd2dbca(tcoil_esd_abcd)
    '''

    pre_tia_channel_dbca = series( pad_dbca, bump_rx_dbca, tline_dbca, bump_tx_dbca, pd_dbca)
    Z_tia_precede = pre_tia_channel_dbca[:,0,0] / pre_tia_channel_dbca[:,1,0]

    tia_abcd, S_tia_noise, tia_trans_impedance, sigma_squared_tia_beforeEQ = channel_model.tia(gm=gm, Rf=Rfb, Ca=25e-15, ft=320e9, Ztot_precede=Z_tia_precede)
    
    #%% channel ABCD
    channel_abcd = series(pd_abcd, bump_tx_abcd, tline_abcd, bump_rx_abcd, pad_abcd, tia_abcd)

    #%% channel frequency response
    H = channel_model.abcd2H(channel_abcd)

    #%% channel impulse response
    h,t = channel_model.H2h(H)

    #%% channel pulse response 
    pulse_response = channel_model.h2pulse(h)

    #%% laser noise calculation at output of tia, before EQ:
    H_laser_noise, f_enbw = channel_model.noise_filter(freq_pole=40e9, plot=False)
    S_laser_noise = channel_model.noise_laser(f_enbw, target_snr_db=40, mean_noise=0)
    H_laser_noise_beforeEQ = H_laser_noise * H 
    S_laser_noise_beforeEQ = S_laser_noise * np.abs(H_laser_noise_beforeEQ)**2
    sigma_squared_laser_beforeEQ = np.trapz(S_laser_noise_beforeEQ, f)

    #%% total noise sigma before EQ
    sigma_squared_beforeEQ = sigma_squared_laser_beforeEQ + sigma_squared_tia_beforeEQ
    
    #%% total signal power before EQ, averaged
    # https://www.slideserve.com/ion/matched-filtering-and-digital-pulse-amplitude-modulation-pam
    # https://cioffi-group.stanford.edu/doc/book/chap1.pdf equation 1.245
    d = max(abs(pulse_response))/2 # assume half-amplitude signaling
    signal_power_beforeEQ = d**2/3 * (M**2-1)
    
    #%% SNR before EQ
    SNR_beforeEQ = 10*np.log10(signal_power_beforeEQ/sigma_squared_beforeEQ)
    print(f'SNR_beforeEQ: {SNR_beforeEQ}')
    
    #%% sampled channel pulse response at baud-rate
    # we are taking all the points, so actually these two values are ignored anyway
    n_taps_pre_channel = 0 
    n_taps_prost_channel = 0
    window = [i for i, e in enumerate(pulse_response) if e != 0] # window that extracts the pulse  
    window_start, window_end = window[0], window[-1]
    pulse_response_afe = np.copy(pulse_response)
    pulse_response = pulse_response[window_start : window_end]
    main_idx=np.argmax(abs(pulse_response))
    sampled_channel_coefficients = channel_model.channel_coefficients(pulse_response, main_idx, n_taps_pre_channel, n_taps_prost_channel, plot=False, all=True)
    
    #%% ISI before EQ:
    ISI_squared_beforeEQ = np.sum(sampled_channel_coefficients**2) - np.max(abs(sampled_channel_coefficients))**2

    #%% total jitter before the EQ:
    mu_n = np.diff(sampled_channel_coefficients)
    jitter_variance = (sigma_jitter * samples_per_symbol)**2 * sum(mu_n)**2
    
    #%% FFE
    n_taps_pre = 2 # they will be updated soon by finding out the best tap delay
    n_taps_post = 27
    n_taps_ffe = n_taps_pre + n_taps_post + 1
    n_taps_dfe = 4
    # dfe_limit = np.array([0.0])   
    
    # print('This is my FFE code')
    ffe = FFE(sampled_channel_coefficients, n_taps_pre, n_taps_post, n_taps_dfe, samples_per_symbol)
    tap_weights_ffe = ffe.mmse_Hossain(SNR=SNR_beforeEQ, signal_power=signal_power_beforeEQ, optimize_delay=True, zf=zf)
    h_ffe = ffe.convolution(tap_weights_ffe, h)
    dfseSNR = ffe.unbiased_SNR
    delay_opt = ffe.n_taps_pre
    
    # print(tap_weights_ffe)
    
    # impulse to pulse
    pulse_response_ffe = channel_model.h2pulse(h_ffe)
    main_idx_ffe = np.argmax(abs(pulse_response_ffe))

    #%% tia noise propagation at the output of FFE
    H_ffe = 0
    for i in range(len(tap_weights_ffe)):
        H_ffe += tap_weights_ffe[i]*np.exp(i*-s/data_rate)

    S_tia_noise_output = S_tia_noise * np.abs(H_ffe)**2
    
    sigma_thermal_squared = np.trapz(S_tia_noise_output, f)
    
    #%% totall frequency response of the channel after EQ.
    H_eq_output = H * H_ffe    

    #%% laser noise calculation
    H_laser_noise_output = H_laser_noise * H_eq_output
    S_laser_noise_output = S_laser_noise * np.abs(H_laser_noise_output)**2
    sigma_laser_squared = np.trapz(S_laser_noise_output, f)
    
    #%% total (thermal) noise
    sigma_noise = np.sqrt(sigma_thermal_squared + sigma_laser_squared)

    '''
    # alternatively, you can do the autocorrelation way to find out total noise:    
    Rnn = np.identity((n_taps_ffe)) * sigma_squared_beforeEQ
    sigma_noise = np.sqrt(tap_weights_ffe.reshape(-1,1).T @ Rnn @ tap_weights_ffe.reshape(-1,1) )
    sigma_noise = sigma_noise[0][0]
    '''

    #%% DFE, assume it is noiseless
    dfe_sampled_channel_coefficients = channel_model.channel_coefficients(pulse_response_ffe, main_idx_ffe, 0, n_taps_dfe, plot=False)
    dfe = DFE(dfe_sampled_channel_coefficients, n_taps_dfe, samples_per_symbol)
    tap_weights_dfe = dfe.coefficients()
    pulse_response_ffe_dfe = dfe.eqaulization(tap_weights_dfe, pulse_response_ffe)

    #%% calculate sigma_isi_squared
    t_sp = np.argmax(np.abs(pulse_response_ffe)) # why not dfe? because dfe may shift some post cursour that has magnitude larger than the main cursor
    
    # calculate residual ISI
    delta_post_isi_list = []
    delta_pre_isi_list = []
    
    for n in range(1,num_symbols):
        # post ISI
        if t_sp+n*samples_per_symbol < len(pulse_response_ffe_dfe):
            if pulse_response_ffe_dfe[t_sp+n*samples_per_symbol] != 0:
                delta_post_isi_list.append(pulse_response_ffe_dfe[t_sp+n*samples_per_symbol])
        # pre ISI
        if t_sp-n*samples_per_symbol > 0:
            if pulse_response_ffe_dfe[t_sp-n*samples_per_symbol] != 0:
                delta_pre_isi_list.append(pulse_response_ffe_dfe[t_sp-n*samples_per_symbol])
    
    sigma_isi_squared = np.sum(np.array(delta_post_isi_list)**2) + np.sum(np.array(delta_pre_isi_list)**2) 
    
    #%% calculate final amplitude of signal 
    A_signal = pulse_response_ffe_dfe[t_sp]
    
    #%% FoM
    FoM = 10*np.log10((A_signal)**2/(sigma_noise**2+sigma_isi_squared+jitter_variance))
    
    
    return {
                'FoM': FoM,
                'A_signal': A_signal,
                'pulse_response': pulse_response_afe,
                'pulse_response_ffe': pulse_response_ffe,
                'pulse_response_ffe_dfe': pulse_response_ffe_dfe,
                'tia_trans_impedance': tia_trans_impedance,
                'H_tia_output': H,
                'H_eq_output': H_eq_output,
                'sigma_thermal_squared': sigma_thermal_squared,
                'sigma_isi_squared': sigma_isi_squared,
                'sigma_laser_squared': sigma_laser_squared,
                'sigma_noise': sigma_noise,
                'tap_weights_ffe': tap_weights_ffe,
                'tap_weights_dfe': tap_weights_dfe,
                'dfseSNR': dfseSNR,
                'delay_opt': delay_opt,
                'sigma_squared_laser_beforeEQ':sigma_squared_laser_beforeEQ,
                'sigma_squared_tia_beforeEQ':sigma_squared_tia_beforeEQ,
                'ISI_squared_beforeEQ': ISI_squared_beforeEQ,
                'SNR_beforeEQ': SNR_beforeEQ,
                'jitter_variance': jitter_variance,
                'h': h,
                'h_ffe': h_ffe,
                'sampled_channel_coefficients': sampled_channel_coefficients
                }

#%%####################### GA implementation #########################
# import pymoo
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_termination, get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.model.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

class MyProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var=3, 
                         n_obj=1, 
                         n_constr=0, 
                         xl =  np.array([3.0, 10, 100.0]), # [tline_width, gm, Rfb], we remove t-coil parameters again due to NDA
                         xu = np.array([20.0, 120, 4000.0]),
                         elementwise_evaluation=True,
                         **kwargs)
        
        
    def _evaluate(self, X, out, *args, **kwargs):
        results = channel_wrapper(channel_model, X)
        
        def f1(action):
            # maximize FoM
            FoM = results['FoM']
            print(f'FoM: {FoM}')
            return -FoM # minimize -FOM = maximize FOM
        
        print(f'action: {X}')

        out['F'] = [f1(X)]
        
        
mask = ['int', 'real', 'real']

sampling = MixedVariableSampling(mask, {
    'real': get_sampling('real_random'),
    'int':get_sampling('int_random')
    })

crossover = MixedVariableCrossover(mask, {
    'real': get_crossover('real_sbx', prob=0.9, eta=20.0),
    'int':get_crossover('int_sbx', prob=0.9, eta=20.0)
    })

mutation = MixedVariableMutation(mask, {
    'real': get_mutation('real_pm', eta=20.0),
    'int': get_mutation('int_pm', eta=20.0)
    })
    

# number of proccess to be used
n_proccess = 8
# initialize the pool
pool = multiprocessing.Pool(n_proccess)

# setup algorithm
algorithm = NSGA2(
    pop_size=100,
    n_offspring=10,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=False # True trigger a bug...
)

# problem = MyProblem(parallelization = ('starmap', pool.starmap))
problem = MyProblem()
termination = get_termination("n_gen", 20)
    

res = minimize(problem, algorithm, termination = termination, seed=1, 
               save_history=True, verbose=True)


print('Processes (sec):', res.exec_time)

pool.close()
pool.join()

#%% final results

X, F = res.opt.get("X", "F")
X = X[0]
zf = False
opt_results = channel_wrapper(channel_model, X, zf=zf)

FoM_opt = opt_results['FoM']
A_signal = opt_results['A_signal']
channel_pulse_response = opt_results['pulse_response']
channel_pulse_response_ffe = opt_results['pulse_response_ffe']
channel_pulse_response_opt = opt_results['pulse_response_ffe_dfe']
H_tia_output =  opt_results['H_tia_output']
H_eq_output =  opt_results['H_eq_output']
tia_trans_impedance = opt_results['tia_trans_impedance']
sigma_thermal_squared = opt_results['sigma_thermal_squared']
sigma_isi_squared = opt_results['sigma_isi_squared']
sigma_laser_squared = opt_results['sigma_laser_squared']
sigma_noise = opt_results['sigma_noise']
tap_weights_ffe = opt_results['tap_weights_ffe']
tap_weights_dfe = opt_results['tap_weights_dfe']
dfseSNR = opt_results['dfseSNR']
delay_opt = opt_results['delay_opt']
ISI_squared_beforeEQ = opt_results['ISI_squared_beforeEQ']
sigma_squared_tia_beforeEQ = opt_results['sigma_squared_tia_beforeEQ']
sigma_squared_laser_beforeEQ = opt_results['sigma_squared_laser_beforeEQ']
SNR_beforeEQ = opt_results['SNR_beforeEQ']
h = opt_results['h']
h_ffe = opt_results['h_ffe']
sampled_channel_coefficients = opt_results['sampled_channel_coefficients']
np.savetxt(f"h_afe.csv", h, delimiter=",")
np.savetxt(f"tap_weights_ffe_zf={zf}.csv", tap_weights_ffe, delimiter=",")
np.savetxt(f"sampled_channel_coefficients_afe.csv", sampled_channel_coefficients, delimiter=",")


sigma_tia_diff = sigma_thermal_squared - sigma_squared_tia_beforeEQ
sigma_isi_diff = sigma_isi_squared - ISI_squared_beforeEQ
sigma_laser_diff = sigma_laser_squared - sigma_squared_laser_beforeEQ
print(f'sigma_tia_diff: {sigma_tia_diff} | sigma_laser_diff: {sigma_laser_diff} | sigma_isi_diff: {sigma_isi_diff}')
print(f'total: {sigma_tia_diff + sigma_isi_diff + sigma_laser_diff}')

H_tia_dB = 20*np.log10(abs(tia_trans_impedance))
plt.title("TIA trans-impedance")
plt.plot(f,H_tia_dB,linewidth=2)
plt.xlim(left=1e9)
plt.xscale('log')
plt.xlabel('Frequency (Hz)', weight='bold')
plt.ylabel('$\mathbf{{trans-impedance} (dB)}$', weight='bold')
plt.grid()
plt.show()
tia_3dB = np.where(np.diff(np.signbit(H_tia_dB - H_tia_dB[0] + 3)))[0] * 1.024e9

H_tia_output_dB = 20*np.log10(abs(H_tia_output))
plt.title("Channel frequency response at output of AFE")
plt.plot(f,H_tia_output_dB,linewidth=2)
plt.xlim(left=1e9)
plt.xscale('log')
plt.xlabel('Frequency (Hz)', weight='bold')
plt.ylabel('$\mathbf{{|H_{output,TIA}|} (dB)}$', weight='bold')
plt.grid()
plt.show()
H_tia_output_3dB = np.where(np.diff(np.signbit(H_tia_output_dB - H_tia_output_dB[0] + 3)))[0] * 1.024e9

H_eq_output_dB = 20*np.log10(abs(H_eq_output))
plt.title("Channel frequency response at output of FFE")
plt.plot(f,H_eq_output_dB,linewidth=2)
plt.xlim(left=1e9)
plt.xscale('log')
plt.xlabel('Frequency (Hz)', weight='bold')
plt.ylabel('$\mathbf{{|H_{output,EQ}|} (dB)}$', weight='bold')
plt.grid()
plt.show()
H_eq_output_3dB = np.where(np.diff(np.signbit(H_eq_output_dB - H_eq_output_dB[0] + 3)))[0] * 1.024e9


#%% check the convergence
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = -np.array([e.opt[0].F for e in res.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--", linewidth=2)
plt.xlabel('simulation', weight='bold')
plt.ylabel('FoM (dB)', weight='bold')
plt.grid()
plt.show()

#%% plot pulse response and eye diagram

pre_cursor = 3
post_cursor = 20

idx_main = np.argmax(abs(channel_pulse_response))
idx_main_ffe = np.argmax(abs(channel_pulse_response_ffe))
idx_main_opt = np.argmax(abs(channel_pulse_response_opt))

channel_model.channel_coefficients(channel_pulse_response, idx_main, pre_cursor, post_cursor, plot=True, title='Pulse Response After AFE')
channel_model.channel_coefficients(channel_pulse_response_ffe, idx_main_ffe, pre_cursor, post_cursor, plot=True, title='Pulse Response After FFE')
channel_model.channel_coefficients(channel_pulse_response_opt, idx_main_opt, pre_cursor, post_cursor, plot=True, title='Pulse Response After DFE')

_ = statistical_eye(pulse_response=channel_pulse_response_opt, 
                                        samples_per_symbol=samples_per_symbol, 
                                        A_window_multiplier=2,
                                        sigma_noise=sigma_noise, 
                                        M=4, 
                                        sample_size=32, 
                                        target_BER=2.4e-4,
                                        upsampling = 16,
                                        mu_jitter=0, # in terms of UI
                                        plot=True, 
                                        noise_flag=True, 
                                        jitter_flag=True)
