import numpy as np
from utils import *


class TIA(object):
    
    def __init__(self, gm, Rf, Ca, ft, f):
    
        self.gm = gm
        self.Rf = Rf
        self.Ca = Ca
        self.ft = ft
        self.f = f
        if self.f[0] == 0:
            self.f[0] = 1 # avoid divided by 0
        self.multiplier = 2.0/3.0 # short channel model
        self.Ra = 6/self.gm
        self.Cg = self.gm/(2*np.pi*self.ft)
        self.Cf = (1-self.multiplier)*self.Cg
        self.Cin = self.multiplier*self.Cg
        self.s = 1j*2*np.pi*self.f
        # print(f'{self.Cg} | {self.Ca} | {self.Cf} | {self.Cin} | {self.Ra}')

    def abcd(self):
        ABCD = np.zeros((len(self.f),2,2), dtype=np.cfloat)
        A = (self.Ra + self.Rf + self.Ca*self.Ra*self.Rf*self.s + self.Cf*self.Ra*self.Rf*self.s)/(self.Ra*(self.Cf*self.Rf*self.s - self.Rf*self.gm + 1))
        B = -1/(self.gm - (self.Cf*self.Rf*self.s + 1)/self.Rf)
        C = ((self.Cf*self.Rf*self.s + 1)*(self.Ra*self.gm + self.Ca*self.Ra*self.s + 1))/(self.Ra*(self.Cf*self.Rf*self.s - self.Rf*self.gm + 1)) + (self.Cin*self.s*(self.Ra + self.Rf + self.Ca*self.Ra*self.Rf*self.s + self.Cf*self.Ra*self.Rf*self.s))/(self.Ra*(self.Cf*self.Rf*self.s - self.Rf*self.gm + 1))
        D = (self.Cf*self.Rf*self.s + self.Cin*self.Rf*self.s + 1)/(self.Cf*self.Rf*self.s - self.Rf*self.gm + 1)
    
        ABCD[:,0,0] = A
        ABCD[:,0,1] = B
        ABCD[:,1,0] = C # 1/C is the transimpedance 
        ABCD[:,1,1] = D
    
        return ABCD

    def tia_output_noise_psd(self, Ztot_precede=None):
        k = 1.38064852e-23 
        T = 300
        gamma=2
        In_gm_squard = 4*k*T*gamma*self.gm
        In_Rf_squard = 4*k*T/self.Rf
       
        Z_Cin = 1/(self.s*self.Cin) 
        
        try:
            if Ztot_precede == None:
                Ztot_precede = Z_Cin # seeing Cgs as the input impedance
        except:
            Ztot_precede = (Ztot_precede * Z_Cin) / (Ztot_precede + Z_Cin)

        Zf = (1/(self.s*self.Cf)*self.Rf)/(self.Rf+1/(self.s*self.Cf))
        Zf[0] = self.Rf
        Za = (1/(self.s*self.Ca)*self.Ra)/(self.Ra+1/(self.s*self.Ca))
        Za[0] = self.Ra
                   
        Znoise_Rf_squared = np.abs((Za*Zf*(1+self.gm*Ztot_precede))/(Zf+Ztot_precede+(1+self.gm*Ztot_precede)*Za))**2 
        Znoise_gm_squared = np.abs((Za*(Zf+Ztot_precede))/(Zf+Ztot_precede+Za*(1+self.gm*Ztot_precede)))**2
        
        S_out = In_Rf_squard * Znoise_Rf_squared + In_gm_squard * Znoise_gm_squared
        
        return S_out

def photo_detector(Cpd, Rpd, f):

    junctioncap = admittance2abcd(1j*2*np.pi*f*Cpd)
    seriesres = impedance2abcd(Rpd*np.ones(len(f)))
    ABCD = series(junctioncap, seriesres)

    return ABCD

def bump(Lseries, Cshunt, f, mode):

    Lseries_abcd = impedance2abcd(1j*2*np.pi*f*Lseries)
    Cshunt_abcd = admittance2abcd(1j*2*np.pi*f*Cshunt)
    bump_tx_abcd = series(Lseries_abcd, Cshunt_abcd)

    if mode == 'tx':
        bump_tx_abcd = series(Lseries_abcd, Cshunt_abcd)

        return bump_tx_abcd

    elif mode == 'rx':
        bump_rx_abcd = series(Cshunt_abcd, Lseries_abcd)

        return bump_rx_abcd
        
    else:
        print('mode can only be "rx" and "tx".')

def tline(TLine, TLine_length, TLine_width, z0=50, dataset='old'):

    Nw = round((TLine_width - 15e-6)/(5e-6)+1)
    d=TLine_length
    f = TLine[:,0]

    if dataset=='old':
    
        s11_real_index = round(0*702+2+((d-1e-3)/(0.5e-3)*18)+Nw-1)
        s11_imag_index = round(1*702+2+((d-1e-3)/(0.5e-3)*18)+Nw-1)
        s12_real_index = round(2*702+2+((d-1e-3)/(0.5e-3)*18)+Nw-1)
        s12_imag_index = round(3*702+2+((d-1e-3)/(0.5e-3)*18)+Nw-1)
    
        s11_TL = TLine[:,s11_real_index-1] + 1j*TLine[:,s11_imag_index-1]
        s12_TL = TLine[:,s12_real_index-1] + 1j*TLine[:,s12_imag_index-1]
    
        TLine_sparams = np.zeros((len(f), 2, 2), dtype=complex)
        
        TLine_sparams[:,0,0] = s11_TL
        TLine_sparams[:,0,1] = s12_TL
        TLine_sparams[:,1,0] = s12_TL
        TLine_sparams[:,1,1] = s11_TL
    
        TLine_abcd = s2abcd(TLine_sparams, f, z0)
    
        return TLine_abcd, f

    if dataset=='new':
        S11_real_index = 0*18+Nw+1
        S11_imag_index = 1*18+Nw+1
        S12_real_index = 2*18+Nw+1
        S12_imag_index = 3*18+Nw+1
        
        S11_TL =  TLine[:,S11_real_index-1] + 1j*TLine[:,S11_imag_index-1]
        S12_TL = TLine[:,S12_real_index-1] + 1j*TLine[:,S12_imag_index-1]
        
        TLine_sparams = np.zeros((len(f), 2, 2), dtype=complex)
        
        TLine_sparams[:,0,0] = S11_TL
        TLine_sparams[:,0,1] = S12_TL
        TLine_sparams[:,1,0] = S12_TL
        TLine_sparams[:,1,1] = S11_TL

        TLine_abcd = s2abcd(TLine_sparams, f, z0)

        return TLine_abcd, f


def tcoil(L, Cesd, k, f, Cb=1e-15):
    '''
    This is just a simplified symmetric t-coil math model, not realistic, but you can use it anyway if you want

    '''

    if f[0]==0:
       f[0] = 1 # avoid divided by zero

    w = 2*np.pi*f
    M = k*L

    Zl = 1j*w*L
    Zb = 1/(1j*w*Cb)
    Zcesd = 1/(1j*w*Cesd)

    Zesd = 1j*w*M + Zcesd

    T_coil_ABCD = np.zeros((len(f), 2, 2), dtype=complex)

    T_coil_ABCD[:,0,0] = (((Zl+Zb)*(Zl+Zesd)+Zesd*Zl))/(Zl*(Zl+Zesd)+(Zb+Zl)*Zesd)
    T_coil_ABCD[:,0,1] = ((Zb*Zl+Zesd*Zb)*(Zl+Zesd)-Zesd**2*Zb)/(Zl*(Zl+Zesd)+(Zb+Zl)*Zesd)
    T_coil_ABCD[:,1,0] = ((Zb+Zl)**2-Zl**2)/((Zl*Zb*(Zl+Zesd)+Zesd*Zb*(Zb+Zl)))
    T_coil_ABCD[:,1,1] = (Zl*(Zb+Zl)+Zesd*(Zb+Zl)+Zesd*Zl)/(Zl*(Zl+Zesd)+Zesd*(Zb+Zl))

    return T_coil_ABCD














