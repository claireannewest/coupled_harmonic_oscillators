import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

load = np.loadtxt('auJC.tab',skiprows=3)
start = 14; end = 29 # change these parameters if you'd like to only fit a part of the wavelength range
wave = load[start:end,0] # Wavelength, assuming in microns
n = load[start:end,1] # These assume your dielectric file has the header "1 2 3 0 0" 
k = load[start:end,2] # These assume your dielectric file has the header "1 2 3 0 0" 
eps_real = n**2 - k**2
eps_imag = 2*n*k
# I collapse the real part and imaginary part of epsilon so I can simultanesouly fit them.
# I.e. optimize.curve_fit doesn't work very well with fitting complex functions, 
eps_rawdata = np.concatenate((eps_real, eps_imag),axis=0) 

scale = 10 # this will scale gamma so all input parameters are around the same order of magnitude
def epsilon_fit(wave, Eps_inf, wp, gam_scale): #returns [eps_real, eps_imag]
	gamma = gam_scale/scale
	w = 1.240/wave
	eps = Eps_inf - wp**2/(w**2+1j*gamma*w)
	return np.concatenate((eps.real, eps.imag), axis=0)

lower_bound = [9, 1, 0]
upper_bound = [10, 10, 1]
bound_array = (lower_bound, upper_bound)

params, params_covariance = optimize.curve_fit(epsilon_fit, wave, eps_rawdata,bounds=bound_array)
print('Fit Parameters:')
print('eps_inf = %.3f' % params[0])
print('wp =  %.3f' %  params[1])
print('gamma =  %.3f' %  (params[2]/scale))

def plot_fit():
	''' Plot dielectric function with raw and fit drude parameters'''
	drude_fit = epsilon_fit(wave=wave, Eps_inf=params[0], wp=params[1], gam_scale=params[2])
	plt.subplot(211)
	plt.plot(wave, eps_rawdata[:int(len(eps_rawdata)/2)], label='JC Data')
	plt.plot(wave, drude_fit[:int(len(eps_rawdata)/2)], label='Fit')
	plt.legend()

	plt.subplot(212)
	plt.plot(wave, eps_rawdata[int(len(eps_rawdata)/2):], label='JC Data')
	plt.plot(wave, drude_fit[int(len(eps_rawdata)/2):], label='Fit')

	plt.show()

def write_fit_dda():
	'''Write dielectric file with new fit parameters'''
	eps_fit = epsilon_fit(wave=wave, Eps_inf=params[0], wp=params[1], gam_scale=params[2])
	eps_realFit = eps_fit[:int(len(eps_rawdata)/2)]
	eps_imagFit = eps_fit[int(len(eps_rawdata)/2):]
	file = open(str('auDrudeFit.tab'),'w')
	file.write(str('Drude mode fit to JC with ') + str('eps_inf = ') + str(np.round(params[0],3) ) + 
		str(' w_p = ') + str(np.round(params[1],3) ) + str(' gam = ') + str(np.round(params[2]/scale,3)) +'\n')
	file.write(str('1 0 0 2 3 = Specifies Epsilon') + '\n')
	file.write(str('lambda	Re[eps] 	Img[eps]') + '\n')
	for j in range(0, len(wave)):
		file.write(str(wave[j]) + '\t' + str(eps_realFit[j]) + '\t' + str(eps_imagFit[j]) + '\n')
	file.close()

def write_fit_bem():
	'''Write dielectric file with new fit parameters'''
	eps_fit = epsilon_fit(wave=wave, Eps_inf=params[0], wp=params[1], gam_scale=params[2])
	eps_realFit = eps_fit[:int(len(eps_rawdata)/2)]
	eps_imagFit = eps_fit[int(len(eps_rawdata)/2):]
	file = open(str('/Users/clairewest/werk/research/MNPBEM17/Material/@epstable/auDrudeFit.dat'),'w')
	file.write(str('% Drude mode fit to JC with ') + str('eps_inf = ') + str(np.round(params[0],3) ) + 
		str(' w_p = ') + str(np.round(params[1],3) ) + str(' gam = ') + str(np.round(params[2]/scale,3)) +'\n')
	file.write(str('% Energy [eV]	n 	k') + '\n')
	eps = eps_realFit + 1j*eps_imagFit
	n = np.sqrt((np.abs(eps) + np.real(eps))/2)
	k = np.sqrt((np.abs(eps) - np.real(eps))/2)
	energy = 1.240/wave
	for j in range(0, len(wave)):
		file.write(str(energy[::-1][j]) + '\t' + str(n[::-1][j]) + '\t' + str(k[::-1][j]) + '\n')
	file.close()

write_fit_bem()
