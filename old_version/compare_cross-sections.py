import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn

e = 4.80326E-10 #statC
c = 2.998E+10 #cm/s
hbar_eVs = 6.58212E-16 #eV*s
hbar_cgs = 1.0545716E-27 
eps_b = 1
wp = 9. # [eV], bulk plasma frequency 
eps_inf = 9.7 # [unitless], static dielectric response of ionic background 

radii = 5.E-7
w = np.arange(2,3.,.001)
D = 1.
li = radii
v = 4./3*np.pi*(radii)**3 # [cm^3], volume of sphere
Li = 1./3
w0_qs = np.sqrt(wp**2/(eps_inf+2*eps_b)) # [eV], quasi-static plasmon resonance frequency 
m = 4.*np.pi*e**2*((eps_inf - eps_b)+1/Li)/((w0_qs/hbar_eVs)**2*v/Li) # [g], quasi-staticc mass of sphere

w0 = np.sqrt(wp**2/(eps_inf+2*eps_b))
print(w0)
gamNR = 0.07
gamR = 2.*e**2*(w*hbar_eVs)**2/(3.*m*c)
gamTot = gamNR + gamR

def alpha():
	return e**2/m * hbar_eVs**2 / (- w**2 - 1j*gamTot*w + w0**2)

def abs_cross():
	return 4.*np.pi*w/c*(gamNR/gamTot)*np.imag(alpha())

def scat_cross():
	return 4.*np.pi*w/c*(gamR/gamTot)*np.imag(alpha())

def ext_cross():
	return 4.*np.pi*w/c*np.imag(alpha())

def gammaEELS(
    ebeam_loc,
    ):     
    v = 0.48 
    gamL = 1/np.sqrt(1-v**2)
    rod_loc = np.array([0,0])
    magR = np.linalg.norm(ebeam_loc-rod_loc)
    constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*(w/hbar_eVs)**2*(kn(1,(w/hbar_eVs)*magR/(v*c*gamL)))**2
    Gam_EELS = constants*np.imag(alpha())
    return Gam_EELS

idx = np.where(abs_cross() == max(abs_cross())); wave_abs = w[idx]
print('Absorption', wave_abs[0])

# idx = np.where(scat_cross() == max(scat_cross())); wave_scat = w[idx]
# print('Scattering', wave_scat[0])

# idx = np.where(ext_cross() == max(ext_cross())); wave_ext = w[idx]
# print('Extinction', wave_ext[0])

# idx = np.where(gammaEELS() == max(gammaEELS())); wave_eel = w[idx]
# print('Gamma eel', wave_eel[0])

plt.plot(w, abs_cross()/max(abs_cross()), label='abs')
# plt.plot(w, scat_cross()/max(scat_cross()), label='scat')
# plt.plot(w, ext_cross()/max(ext_cross()), label='ext')
# plt.plot(w, gammaEELS(np.array([-46*2*1E-7,0])), label='eel')
plt.legend()
plt.show()