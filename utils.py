"""Functions for Channel Modelling

Notes
-----
Based off of https://github.com/tchancarusone/Wireline-ChModel-Matlab
"""

#TODO: change comments to docstring

import numpy as np
import matplotlib.pyplot as plt

def abcd2dbca(ABCD):
    
    A = ABCD[:,0,0]
    B = ABCD[:,0,1]
    C = ABCD[:,1,0]   
    D = ABCD[:,1,1] 
    
    DBCA = np.copy(ABCD)

    DBCA[:,0,0] = D
    DBCA[:,0,1] = B   
    DBCA[:,1,0] = C
    DBCA[:,1,1] = A
    
    return DBCA
    

def rlgc2abcd(r,l,g,c,d,f):
    """

    Parameters
    ----------
    r : TYPE
        r = resistance along the line per unit length [?/m]
    l : TYPE
        inductance along the line per unit length [H/m]
    g : TYPE
        conductance shunting the line per unit length [S/m]
    c : TYPE
        capacitance shunting the line per unit length [F/m]
    d : TYPE
        transmissioin line length
    f : TYPE
        frequency, row vector.

    Returns
    -------
    ABCD : TYPE
        ABCD matrix.

    """
    
    w = 2*np.pi*f
    gammad = d*np.sqrt( np.multiply( (r+1j * np.multiply(w,l)),(g+ 1j * np.multiply(w,c)) ) );
    z0 = np.sqrt( np.divide((r + 1j*np.multiply(w,l) ),(g+1j*np.multiply(w,c)))  );
    A = np.cosh(gammad)
    B = np.multiply(z0,np.sinh(gammad))
    C = np.divide(np.sinh(gammad),z0)
    D = A
    
    ABCD = np.zeros((f.size,2,2),dtype=np.complex_)
    ABCD[:,0,0] = A
    ABCD[:,0,1] = B   
    ABCD[:,1,0] = C
    ABCD[:,1,1] = D
    
    return ABCD

def impedance2abcd(z):
    """
    

    Parameters
    ----------
    z : TYPE
        2-port network with series impedance z (z is array if frequency dependent)

    Returns
    -------
    ABCD : TYPE
        ABCD matrix

    """
    
    l = z.size
    ABCD = np.zeros((l,2,2),dtype=np.complex_)
    ABCD[:,0,0] = np.ones(l)
    ABCD[:,0,1] = z   
    ABCD[:,1,0] = np.zeros(l)
    ABCD[:,1,1] = np.ones(l)
    return ABCD
 
def admittance2abcd(y):
    """
    

    Parameters
    ----------
    y : TYPE
        for 2-port network with shunt admittance y (y is array if frequency dependent)

    Returns
    -------
    ABCD : TYPE
        ABCD matrix.

    """
    
    l = y.size
    ABCD = np.zeros((l,2,2),dtype=np.complex_)
    ABCD[:,0,0] = np.ones(l)
    ABCD[:,0,1] = np.zeros(l)
    ABCD[:,1,0] = y
    ABCD[:,1,1] = np.ones(l)
    return ABCD

def series(ABCD1, ABCD2, *arg):
    """
    Series combination of two or more 2-port networks, ABCD1 followed by ABCD2

    Parameters
    ----------
    ABCD1: TYPE
        ABCD matrix 1.
    ABCD2: TYPE
        ABCD matrix 2 following ABCD matrix 1.
    *arg: tuples:
        more ABCD matrices following ABCD2

    Returns
    -------
    ABCD_cascade: TYPE
        Cascading ABCD matrix of ABCD1 and ABCD2

    """
    
    ABCD_cascade = ABCD1 @ ABCD2
    if len(arg) != 0:
        for i in range(len(arg)):
            ABCD_cascade = ABCD_cascade @ arg[i]
            
    return ABCD_cascade


def freq2impulse(H, f):
    """
    Returns the impulse response, h, and (optionally) the step response,
    hstep, for a system with complex frequency response stored in the array H
    and corresponding frequency vector f.  The time array is
    returned in t.  The frequency array must be linearly spaced.

    Parameters
    ----------
    H : TYPE
        (complex) frequency response.
    f : TYPE
        frequency.

    Returns
    -------
    h : TYPE
        impluse response.
    t : TYPE
        time vector.

    """
    
    Hd = np.concatenate((H,np.conj(np.flip(H[1:H.size-1]))))
    h = np.real(np.fft.ifft(Hd))
    hstep = np.convolve(h,np.ones(h.size))
    hstep = hstep[0:h.size]
    t= np.linspace(0,1/f[1],h.size+1)
    t = t[0:-1]
    
    return h,t,hstep
        
def s2abcd(sparams,f, z0=50):
    """
    ABCD matrix description of a 2-port network with S-parameters
    specified at the frequencies f in row vectors s11,s12,s21,s22
    
    f should be a row vector
    
    z0 is the characteristic impedance used for the S-parameter
    measurements
    
    Returns a structure containing the 2-port A,B,C,D matrix entries
    at the frequencies in f: s.A, s.B, s.C, s.D

    Parameters
    ----------
    sparams : TYPE
        sparams.
    f : TYPE
        frequency.
    z0 : TYPE
        characteristic impedance.

    Returns
    -------
    ABCD : TYPE
        ABCD matrix.

    """
    
    ABCD = np.zeros((f.size,2,2),dtype=np.complex_)
    
    s11 = sparams[:,0,0]
    s12 = sparams[:,0,1]
    s21 = sparams[:,1,0]
    s22 = sparams[:,1,1]
    
    ABCD[:,0,0] = ((1+s11)*(1-s22) + s12*s21) / (2*s21)
    ABCD[:,0,1] = z0 * ((1+s11)*(1+s22) - s12*s21) / (2*s21)
    ABCD[:,1,0] = (1/z0) * ((1-s11)*(1-s22) - s12*s21) / (2*s21)
    ABCD[:,1,1] = ((1-s11)*(1+s22) + s12*s21) / (2*s21)

    return ABCD    

def abcd2s(ABCD, f, z0=50):
    
    sparams = np.zeros((f.size,2,2),dtype=np.complex_)

    A = ABCD[:,0,0]
    B = ABCD[:,0,1]
    C = ABCD[:,1,0]
    D = ABCD[:,1,1]
    
    s11 = (A + B/z0 - C*z0 - D) / (A + B/z0 + C*z0 + D)
    s12 = 2*(A*D-B*C) / (A + B/z0 + C*z0 + D)
    s21 = 2/(A + B/z0 + C*z0 + D)
    s22 = (-A + B/z0 - C*z0 + D) / (A + B/z0 + C*z0 + D)
    
    sparams[:,0,0] = s11
    sparams[:,0,1] = s12
    sparams[:,1,0] = s21
    sparams[:,1,1] = s22
    
    return sparams
    
def channel_coefficients(pulse_response, main_idx, t, samples_per_symbol, n_precursors, n_postcursors, title="Channel Coefficients", plot=True, all=False):

    if all == False:
        n_cursors = n_precursors + n_postcursors + 1
        channel_coefficients = np.zeros(n_cursors)
    else:
        n_precursors = 0
        n_postcursors = 0
        while main_idx - samples_per_symbol*n_precursors > 0:
            n_precursors = n_precursors + 1
        while main_idx + samples_per_symbol*n_postcursors < len(pulse_response) - 2*samples_per_symbol:
            n_postcursors = n_postcursors + 1
            
        n_cursors = n_precursors + n_postcursors + 1
        channel_coefficients = np.zeros(n_cursors)        

    t_vec = np.zeros(n_cursors)
    xcoords = []
    half_symbol = int(round(samples_per_symbol/2))
    
    #find peak of pulse response
    # main_idx = np.argmax(abs(pulse_response))
    
    for cursor in range(n_cursors):
        
        a = cursor - n_precursors
        
        channel_coefficients[cursor] = pulse_response[main_idx+a*samples_per_symbol]
        
        #for plotting
        xcoords = xcoords + [1e9*t[main_idx+a*samples_per_symbol-half_symbol]]
        t_vec[a+n_precursors] = t[main_idx + a*samples_per_symbol]
        
    xcoords = xcoords + [1e9*t[main_idx+(n_postcursors+1)*samples_per_symbol-half_symbol]]
    
    if plot==True:
        #plot pulse response and cursor samples
        plt.figure()
        plt.plot(t_vec*1e9, channel_coefficients, 'o', label = 'Cursor samples')
        plt.plot(t*1e9,pulse_response, label = 'Pulse response', linewidth=2)
        plt.xlabel("Time [ns]", weight='bold')
        plt.ylabel("Amplitude [V]", weight='bold')
        
        ll = t[main_idx-samples_per_symbol*(n_precursors+2)]*1e9
        ul = t[main_idx+samples_per_symbol*(n_postcursors+2)]*1e9
        
        #print(ll,ul)
        plt.xlim([ll,ul])
        plt.title(title)
        plt.legend()
        for xc in xcoords:
            plt.axvline(x=xc,color = 'grey',label ='UIs', ls='--')
    
    return channel_coefficients   
    

def pam4_input(samples_per_symbol, data_in, voltage_levels):
    
    """Genterates ideal, square, PAM-4 transmitter waveform from binary sequence
    Parameters
    ----------
    samples_per_symbol: int
        timesteps per bit
    
    length: int
        length of desired time-domain signal
    
    data_in: array
        quaternary sequence to input, must be longer than than length/samples_per_symbol
    
    voltage levels: array
        definition of voltages corresponding to symbols. 
        voltage_levels[0] = voltage corresponding to 0 symbol, 
        voltage_levels[1] = voltage corresponding to 1 symbol
        voltage_levels[2] = voltage corresponding to 2 symbol
        voltage_levels[3] = voltage corresponding to 3 symbol
    
    length: float
        timestep of time domain signal
    
    Returns
    -------
    signal: array
        square waveform at trasmitter corresponding to data_in
    """
    
    signal = np.zeros(samples_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i]==0):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[0]
        elif (data_in[i]==1):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[1]
        elif (data_in[i]==2):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[2]
        elif (data_in[i]==3):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[3]
        else:
            print('unexpected symbol in data_in')
            return False

        if (i%100000 == 0):
            print('i=',i)
    
    return signal


def nrz_input(samples_per_symbol, data_in, voltage_levels):
    
    """Genterates  ideal, square, NRZ (PAM-2) transmitter waveform from binary sequence
    Parameters
    ----------
    samples_per_symbol: int
        timesteps per bit
    
    length: int
        length of desired time-domain signal
    
    data_in: array
        binary sequence to input, must be longer than than length/samples_per_symbol
    
    voltage levels: array
        definition of voltages corresponding to 0 and 1. 
        voltage_levels[0] = voltage corresponding to 0 bit, 
        voltage_levels[1] = voltage corresponding to 1 bit
    
    length: float
        timestep of time domain signal
    
    Returns
    -------
    signal: array
        square waveform at trasmitter corresponding to data_in
    """
    
    signal = np.zeros(samples_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i] == 0):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[0]
        elif(data_in[i] == 1):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol]  = np.ones(samples_per_symbol)*voltage_levels[1]
        else:
            print('unexpected symbol in data_in')
            return False
            
        #if (i%100000 == 0):
         #   print('i=',i)
    
    return signal