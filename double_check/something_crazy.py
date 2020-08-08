import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn

fig = plt.figure(figsize=[4,4])

n = 1.0
radii1 = 20E-7
radii2 = 20E-7
gap = 40E-7

######### SIMULATION #########
##############################
shape_bothstring = str(int(radii1*10**7))+str('nm')+str(int(radii2*10**7))+str('nm_')+str(int(gap*10**7))+str('nm')
shape_onestring = str(int(radii1*10**7))+str('nmsph')
shape_twostring = str(int(radii2*10**7))+str('nmsph')

def plot_onespectra(src, shape, shape_param, diel_data, n):
    filename = str('simulated_spectra/single_sphere/')+str('Spectrum_')+str(src)+str('_')+str(shape)+str(shape_param)+str('_')+str(diel_data)+str('_')+str(n)
    if src == 'bemret' or src == 'mie' or src == 'bemstat':
        data = np.loadtxt(filename,skiprows=1)
        energy = data[:,0]
        abs_cross = data[:,1]
    plt.plot(energy, abs_cross, 'k--',label=str(shape)[:-3]+str(' sim'))

def plot_twospectra(src, shape, diel_data, n):
    filename = str('simulated_spectra/two_spheres/')+str('Spectrum_')+str(src)+str('_')+str(shape)+str('_')+str(diel_data)+str('_')+str(n)
    data = np.loadtxt(filename,skiprows=1)
    energy = data[:,0]
    ext_cross = data[:,1]
    abs_cross = data[:,2]
    sca_cross = data[:,3]

    plt.plot(energy, ext_cross, 'k',alpha=.5,label=str(shape)[:-3]+str('sim'))
    plt.plot(energy, abs_cross, 'b',alpha=.5,label=str(shape)[:-3]+str('sim'))

    # plt.plot(energy, ext_cross/max(ext_cross), label=str(shape)[:-3]+str('sim'))
    # plt.plot(energy, sca_cross/max(sca_cross), label=str(shape)[:-3]+str('sim'))

plot_twospectra(src='bemret', shape=shape_bothstring, diel_data='drude', n=str(n))
# plot_onespectra(src='bemret', shape=shape_onestring, shape_param='144',diel_data='drude', n=str(n))
# plot_onespectra(src='bemret', shape=shape_twostring, shape_param='144',diel_data='drude', n=str(n))

########################################################
# radii = radii1 # [cm]

eps_b = n**2
e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]
wp = 8.959/hbar_eVs # [1/s], bulk plasma frequency 

eps_inf = 9.695 # [unitless], static dielectric response of ionic background 
gamNR_qs = 0.073/hbar_eVs # [1/s]

centers = np.array([ [[ 0, -radii1-gap/2, 0]] , [[0, radii2+gap/2, 0]] ])
unit_vecs = np.array([ [[ 0, 1, 0]] , [[0, 1, 0]] ])
all_radii = np.array([ [[ radii1, radii1, radii1]] , [[radii2, radii2, radii2]] ])

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

def mie_coefficents(w,radius):
    eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
    m = np.sqrt(eps)
    k = w/c
    x = k*radius
    numer_a = m*psi(m*x)*psi_prime(x) - psi(x)*psi_prime(m*x)
    denom_a = m*psi(m*x)*xi_prime(x) - xi(x)*psi_prime(m*x)
    numer_b = psi(m*x)*psi_prime(x) - m*psi(x)*psi_prime(m*x)
    denom_b = psi(m*x)*xi_prime(x) - m*xi(x)*psi_prime(m*x)
    a = numer_a/denom_a
    b = numer_b/denom_b
    return a, b

def alpha0(w,radius): 
    k = w/c
    a, b = mie_coefficents(w,radius)
    alpha0 = 3/(2.*k**3)*1j*(a+b)  
    return alpha0

def A_ij(dip_i, dip_j, k):
    A_ij = np.zeros( (1, 3, 3) ,dtype=complex) 
    r_ij = centers[dip_i,0,:] - centers[dip_j,0,:] # [cm], distance between ith and jth dipole 
    mag_rij = np.linalg.norm(r_ij)
    nhat_ij = r_ij / mag_rij # [unitless], unit vector in r_ij direction.
    # print(nhat_ij)
    # if mag_rij == 0: 
    #     A_ij[0,0,:] = 0 
    # else:
    #     far_field = k**2*mag_rij**2*(1 - nhat_ij*nhat_ij)
    #     intermed = -1j*k*mag_rij*(1 - 3*nhat_ij*nhat_ij)
    #     near_field = (1 - 3*nhat_ij*nhat_ij)
    #     A_ij[0,0,:] = np.exp(1j*k*mag_rij)/mag_rij**3*(far_field + intermed + near_field)
    r = mag_rij
    print(r_ij.shape)
    rx = r_ij[0]; ry = r_ij[1]; rz = r_ij[2]
    A_ij_xx = A_ij[0,0,0]
    A_ij_xy = A_ij[0,0,1]
    A_ij_xz = A_ij[0,0,2]

    A_ij_yx = A_ij[0,1,0]   
    A_ij_yy = A_ij[0,1,1]
    A_ij_yz = A_ij[0,1,2]

    A_ij_zx = A_ij[0,2,0]
    A_ij_zy = A_ij[0,2,1]
    A_ij_zz = A_ij[0,2,2]

    A_ij_xx = np.exp(1j*k*r)/r**3*(k**2*(rx*rx-r**2) + (1-1j*k*r)/r**2*(3*rx*rx-r**2))
    A_ij_xy = np.exp(1j*k*r)/r**3*(k**2*(rx*ry-r**2) + (1-1j*k*r)/r**2*(3*rx*ry-r**2))
    A_ij_xz = np.exp(1j*k*r)/r**3*(k**2*(rx*rz-r**2) + (1-1j*k*r)/r**2*(3*rx*rz-r**2))

    A_ij_yx = A_ij_xy
    A_ij_yy = np.exp(1j*k*r)/r**3*(k**2*(ry*ry-r**2) + (1-1j*k*r)/r**2*(3*ry*ry-r**2))
    A_ij_yz = np.exp(1j*k*r)/r**3*(k**2*(ry*rz-r**2) + (1-1j*k*r)/r**2*(3*ry*rz-r**2))

    A_ij_zx = A_ij_xz
    A_ij_zy = A_ij_yz
    A_ij_zz = np.exp(1j*k*r)/r**3*(k**2*(rz*rz-r**2) + (1-1j*k*r)/r**2*(3*rz*rz-r**2))
    return A_ij

def A(w):
    '''matrix[N, N, 3]'''
    N = 2
    k = w/c
    matrix = np.zeros( (N, N, 3) ,dtype=complex) 
    all_alpha = np.zeros( (N, 1, 3) ,dtype=complex) 
    # Assumes spheres are isotropic 
    all_alpha[0,0,:] = alpha0(w=w, radius=all_radii[0,0,:])
    all_alpha[1,0,:] = alpha0(w=w, radius=all_radii[1,0,:])

    matrix[0,0,:] = all_alpha[0,0,:]**(-1)
    matrix[1,1,:] = all_alpha[1,0,:]**(-1)

    for dip_i in range(0 , N): 
        for dip_j in range(0, N): 
                if dip_i != dip_j:
                    matrix[ dip_i, dip_j, :] = A_ij(dip_i=dip_i, dip_j=dip_j, k=k)
    return matrix

def dda_abs_cross_mie_polariz(w):
    ''' Works for spheres up to 50 nm radii, and in the window < 3.0 eV '''
    k = w/c
    N = 2
    E0 = np.array([ [[ 0, 1, 0]] , [[0, 1, 0]] ])
    A_inv = np.zeros( (N, N, 3) ,dtype=complex)
    P = np.zeros( (N, 1, 3) ,dtype=complex) 

    A_inv[:,:,0] = np.linalg.inv(A(w)[:,:,0])
    A_inv[:,:,1] = np.linalg.inv(A(w)[:,:,1])
    A_inv[:,:,2] = np.linalg.inv(A(w)[:,:,2])

    Px = np.matmul(A_inv[:,:,0], E0[:,:,0])
    Py = np.matmul(A_inv[:,:,1], E0[:,:,1])
    Pz = np.matmul(A_inv[:,:,2], E0[:,:,2])

    P[:,0,0] = np.reshape(Px,(N,))
    P[:,0,1] = np.reshape(Py,(N,))
    P[:,0,2] = np.reshape(Pz,(N,))

    P1 = P[0,0,:]
    E1 = alpha0(w=w, radius=all_radii[0,0,0])**(-1)*P1

    ext_cross_1 = 4*np.pi*k*np.imag( np.sum(np.conj(E1)*P1) )*10**8
    sca_cross_1 = 4*np.pi*k*2/3*k**3*np.sum(P1*np.conj(P1))*10**8
    abs_cross_1 = ext_cross_1 - sca_cross_1

    P2 = P[1,:,:]
    E2 = alpha0(w=w, radius=all_radii[1,0,0])**(-1)*P2
    ext_cross_2 = 4*np.pi*k*np.imag( np.sum(np.conj(E2)*P2) )*10**8
    sca_cross_2 = 4*np.pi*k*2/3*k**3*np.sum(P2*np.conj(P2))*10**8
    abs_cross_2 = ext_cross_2 - sca_cross_2

    Cext = ext_cross_1+ext_cross_2
    Cabs = abs_cross_1+abs_cross_2
    Csca = sca_cross_1+sca_cross_2
    return Cext, Cabs, Csca

# w = np.arange(1.5,3,.005)/hbar_eVs
# dda_ext = np.zeros(len(w))
# dda_abs = np.zeros(len(w))
# dda_sca = np.zeros(len(w))

# for i in range(0,len(w)):
#     dda_ext[i], dda_abs[i], dda_sca[i] = dda_abs_cross_mie_polariz(w=w[i])
# plt.plot(w*hbar_eVs, dda_ext,'k--',label='dda ext')
# plt.plot(w*hbar_eVs, dda_abs,'b--',label='dda abs')

# plt.xlabel('Energy [eV]')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.ylabel('Abs. Cross-section [$\mu$m$^2$]')
# plt.xlim([2.    , 2.75])
# plt.tight_layout()
# plt.legend(frameon=False,loc='upper left')
# plt.show()
