import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn

fig = plt.figure(figsize=[4,4])

### Spectrum_<src>_<shape><shape param>_<diel. data>_<n>

def plot_spectra(src, shape, shape_param, diel_data, n):
	filename = str('simulated_spectra/')+str('Spectrum_')+str(src)+str('_')+str(shape)+str(shape_param)+str('_')+str(diel_data)+str('_')+str(n)
	if src == 'bemret' or src == 'mie' or src == 'bemstat':
		data = np.loadtxt(filename,skiprows=1)
		energy = data[:,0]
		abs_cross = data[:,1]
	else:
		data = np.loadtxt(filename)
		data_sort = data[data[:,1].argsort(),]
		energy = 1.240/data_sort[:,1]
		abs_cross = data_sort[:,3]*(np.pi*data_sort[0,0]**2)
	return energy, abs_cross

n = 1.0
radii = 50.E-7 # [cm]

eps_b = n**2
e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]
wp = 8.959/hbar_eVs # [1/s], bulk plasma frequency 

eps_inf = 9.695 # [unitless], static dielectric response of ionic background 
gamNR_qs = 0.073/hbar_eVs # [1/s]

v = 4./3*np.pi*(radii)**3 # [cm^3], volume of sphere
w0_qs = np.sqrt(wp**2/(eps_inf+2*eps_b)) # [1/s], quasi-static plasmon resonance frequency 
m_qs = 4.*np.pi*e**2/(v*w0_qs**2)*(eps_inf + 2*eps_b)/(3**2*eps_b)

D = 1/3.
m = m_qs + D*e**2/(radii*c**2)
w0 = w0_qs*np.sqrt(m_qs/m)
gamNR = gamNR_qs*(m_qs/m)

def analytic_abs_cross(w):
	gamR = 2*e**2/(3*m*c**3)*w**2
	gam = gamNR + gamR
	abs_cross_analy = 4.*np.pi*e**2/(c*n) * gamNR*w**2/m / ( (-w**2 + w0**2 )**2 + (w*gam)**2) * 10**8 # converts cm^2 to um ^2
	return abs_cross_analy

def dda_abs_cross(w):
	eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
	alpha_CM = 3*v/(4*np.pi)*( (eps - eps_b)/(eps+2*eps_b))
	k = w/c#*np.sqrt(eps_b)
	nextorder = 2*1j/5*(k*radii)**5*(eps**2 - 3*eps + 2)/(eps + 2)**2
	alpha_upd = alpha_CM/(1 - 0*k**2/radii*alpha_CM - 2./3*1j*k**3*alpha_CM - 0*nextorder)
	abs_cross = 1/n*4*np.pi*k*(np.imag(alpha_upd)- 2./3*k**3*np.abs(alpha_upd)**2)*10**8
	return abs_cross

def psi(rho):
	return rho*spherical_jn(1, rho)

def psi_prime(rho):
	return spherical_jn(1, rho) + rho*spherical_jn(1, rho, derivative=True)

def hankel(rho):
	return spherical_jn(1, rho) + 1j*spherical_yn(1, rho)

def hankel_prime(rho):
	return spherical_jn(1, rho, derivative=True) + 1j*spherical_yn(1, rho,derivative=True)

def xi(rho):
	return rho*hankel(rho)

def xi_prime(rho):
	return hankel(rho) + rho*hankel_prime(rho)

def mie_coefficents(w):
	eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
	m = np.sqrt(eps)
	k = w/c
	x = k*radii
	numer_a = m*psi(m*x)*psi_prime(x)-psi(x)*psi_prime(m*x)
	denom_a = m*psi(m*x)*xi_prime(x) - xi(x)*psi_prime(m*x)
	numer_b = psi(m*x)*psi_prime(x)-m*psi(x)*psi_prime(m*x)
	denom_b = psi(m*x)*xi_prime(x) - m*xi(x)*psi_prime(m*x)
	a = numer_a/denom_a
	b = numer_b/denom_b
	return a, b

def my_mie_func(w):
	k = w/c
	a, b = mie_coefficents(w)
	c_sca = (2/(k*radii)**2*(3)*(np.abs(a)**2 + np.abs(b)**2))*np.pi*radii**2
	c_ext = (2/(k*radii)**2*(3)*(np.real(a+b)))*np.pi*radii**2
	c_abs = (c_ext - c_sca)*10**8
	return c_abs

def mie_analy(w):
	eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
	k = w/c#*np.sqrt(eps_b)
	a = -2/3*1j*(eps-1)/(eps+2)*(k*radii)**3 - 1j*2/5*(eps-2)*(eps-1)/(eps+2)**2*(k*radii)**5
	b = -1j/45*(eps-1)*(k*radii)**5
	c_sca = (2/(k*radii)**2*(3)*(np.abs(a)**2 + np.abs(b)**2))*np.pi*radii**2
	c_ext = (2/(k*radii)**2*(3)*(np.real(a+b)))*np.pi*radii**2
	c_abs = (c_ext - c_sca)*10**8
	return c_abs

def dda_abs_cross_mie_polariz(w):
	''' Works for spheres up to 50 nm radii, and in the window < 3.0 eV '''
	eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
	k = w/c
	a, b = mie_coefficents(w)
	alpha_0 = 3/(2*k**3)*1j*(a+b)
	alpha_tilde = alpha_0
	abs_cross = 1/n*4*np.pi*k*(np.imag(alpha_tilde) - 2./3*k**3*np.abs(alpha_tilde)**2)*10**8
	return abs_cross


w = np.arange(1.5,4,.005)/hbar_eVs
osc = np.zeros(len(w))
dda = np.zeros(len(w))
mie = np.zeros(len(w))
my_mie = np.zeros(len(w))
dda_mie = np.zeros(len(w))

for i in range(0,len(w)):
	# print(w)
	osc[i] = analytic_abs_cross(w=w[i])
	dda[i] = dda_abs_cross(w=w[i])
	mie[i] = mie_analy(w=w[i])
	my_mie[i] = my_mie_func(w=w[i])
	dda_mie[i] = dda_abs_cross_mie_polariz(w=w[i])

# shape_string = str(int(radii*10**7))+str('nmsph')

# en_bemdruderet, abs_bemdruderet = plot_spectra(src='bemret', shape=shape_string, shape_param='144', diel_data='drude', n=str(n))

# en_mie, abs_mie = plot_spectra(src='mie', shape=shape_string, shape_param='144', diel_data='drude', n=str(n))

# en_bemdrudestat, abs_bemdrudestat = plot_spectra(src='bemstat', shape=shape_string, shape_param='144', diel_data='drude', n=str(n))

# en_gddadrude, abs_gddadrude = plot_spectra(src='gdda7.3', shape='5nmsph', shape_param='40', diel_data='drude', n=str(n))
# en_gddadrude_100, abs_gddadrude_100 = plot_spectra(src='gdda7.3', shape='5nmsph', shape_param='100', diel_data='drude', n=str(n))

# plt.plot(en_bemdruderet, abs_bemdruderet, c='black',label='bem ret')
# plt.plot(en_mie, abs_mie, c='green',label='bem mie')

# plt.plot(w*hbar_eVs, n**2*dda_mie,label='analytics (dda)')
# plt.plot(w*hbar_eVs, n**2*osc, 'r--',label='analytic (osc)')
# plt.plot(w*hbar_eVs, n**2*mie, 'purple', label='analytic (mie)')
# plt.plot(w*hbar_eVs, n**2*my_mie, linestyle='--',color='pink',label='analytic (mie)')

######################################################################################
######################################################################################
######################################################################################

######### SIMULATION #########
##############################
def plot_onespectra(src, shape, shape_param, diel_data, n):
	filename = str('simulated_spectra/single_sphere/')+str('Spectrum_')+str(src)+str('_')+str(shape)+str(shape_param)+str('_')+str(diel_data)+str('_')+str(n)
	if src == 'bemret' or src == 'mie' or src == 'bemstat':
		data = np.loadtxt(filename,skiprows=1)
		energy = data[:,0]
		abs_cross = data[:,1]
	plt.plot(energy, abs_cross, label=shape)

def plot_twospectra(src, shape, diel_data, n):
	filename = str('simulated_spectra/two_spheres/')+str('Spectrum_')+str(src)+str('_')+str(shape)+str('_')+str(diel_data)+str('_')+str(n)
	data = np.loadtxt(filename,skiprows=1)
	energy = data[:,0]
	ext_cross = data[:,1]
	abs_cross = data[:,2]
	plt.plot(energy, abs_cross, label=shape)

radii1 = 20E-7
radii2 = 30E-7
gap = 30E-7
shape_bothstring = str(int(radii1*10**7))+str('nm')+str(int(radii2*10**7))+str('nm_')+str(int(gap*10**7))+str('nm')
shape_onestring = str(int(radii1*10**7))+str('nmsph')
shape_twostring = str(int(radii2*10**7))+str('nmsph')

plot_twospectra(src='bemret', shape=shape_bothstring, diel_data='drude', n=str(n))
plot_onespectra(src='bemret', shape=shape_onestring, shape_param='144',diel_data='drude', n=str(n))
plot_onespectra(src='bemret', shape=shape_twostring, shape_param='144',diel_data='drude', n=str(n))

############ MODEL ############
###############################






plt.xlabel('Energy [eV]')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ylabel('Abs. Cross-section [$\mu$m$^2$]')
plt.xlim([2, 2.75])
plt.tight_layout()
plt.legend(frameon=False,loc='upper left')
plt.show()



