import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.linalg import null_space
from sympy import *

eps_b = 1.**2 # dielectric constant of background, (eps_b = 1.0 is vacuum) [unitless]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]
hbar_cgs = 1.0545716E-27 # Planck's constant [cm^2*g/s]
e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
prec = 10 # convergence condition for iteration

class AnalyticOscillators:
    def __init__(self, num_part, num_dip, centers, orientations, radii, kind, optional_semiaxis, drive): 
        """Defines the different system parameters.
        
        Keyword arguments:
        num_part -- number of particles 
        num_dip -- number of dipoles per particle (normally two)
        centers -- particle centers [cm]
        orientaitons -- unit vectors defining each dipole [unitless]
        radii -- radii of the oscillators [cm]
        kind -- which kind of dipole (0 = sphere, 1 = long axis prolate spheroid, 2 = short axis prolate spheroid)
        optional_semiaxis -- define the semi-axis of prolate sphereiod if it isn't defined in data file
        """
        self.num_part = num_part
        self.num_dip = num_dip 
        self.radii = radii
        self.centers = centers 
        self.unit_vecs = orientations
        self.kind = kind
        self.optional_semiaxis = optional_semiaxis
        self.drive = drive
        self.w0, self.m, self.gamNR = self.dipole_parameters()
        self.mat_size = int(self.num_dip*self.num_part)
            

    def analytic_twooscill(self,w):
        """Solves for the power absorbed by two coupled oscillators (q.s.) 
        
        Keywords: 
        w -- energy [eV]
        """
        I0 = 10**12 # [erg s^-1 cm^-2] 10^9 W/m^2
        E0 = np.sqrt(I0*8*np.pi/c*np.sqrt(eps_b)) # [g^1/2 s^-1 cm^-1/2]
        dip_i = 0; dip_j = 1
        k = w/hbar_eVs*np.sqrt(eps_b)/c
        w0, m, gamNR = self.dipole_parameters()
        m1 = m[0]; m2 = m[1]; 
        w01 = w0[0]; w02 = w0[1]
        g = np.real(self.coupling(dip_i=dip_i, dip_j=dip_j, k=k)) # off-diagonal matrix elements
        gam = self.gamNR + (w)**2*(2.0*e**2)/(3.0*self.m*c**3)/hbar_eVs # radiative damping for dipole i

        Omega1 = -w**2 - 1j*w*gam[0] + w01**2
        Omega2 = -w**2 - 1j*w*gam[1] + w02**2
        alph1 = np.real(Omega1); alph2 = np.real(Omega2)
        beta1 = np.imag(Omega1); beta2 = np.imag(Omega2)

        a1 = g*m1*m2*(alph1*alph2 - beta1*beta2) + m1*m2**2*alph1*(alph2**2 + beta2**2) - g**2*m2*alph2 - g**3 # [g^3 eV^6]
        b1 = g*m1*m2*(alph2*beta1 + alph1*beta2) + m1*m2**2*beta1*(alph2**2 + beta2**2) + g**2*m2*beta2 # [g^3 eV^6]
        P1 = 1./2*m1*self.gamNR[0]*(e*E0)**2*w**2/(np.abs(m1*m2*Omega1*Omega2 - g**2)**4)*(a1**2 + b1**2)*hbar_eVs # [g cm^2 s^-3]

        a2 = g*m1*m2*(alph1*alph2 - beta1*beta2) + m2*m1**2*alph2*(alph1**2 + beta1**2) - g**2*m1*alph1 - g**3 # [g^3 eV^6]
        b2 = g*m1*m2*(alph2*beta1 + alph1*beta2) + m2*m1**2*beta2*(alph1**2 + beta1**2) + g**2*m1*beta1 # [g^3 eV^6]
        P2 = 1./2*m2*self.gamNR[1]*(e*E0)**2*w**2/(np.abs(m1*m2*Omega1*Omega2 - g**2)**4)*(a2**2 + b2**2)*hbar_eVs # [g cm^2 s^-3]
        return P1, P2

    def plot_analytic_twooscill(self):
        w = np.arange(1,4,.005)
        P1 = np.zeros(len(w))
        P2 = np.zeros(len(w))
        for i in range(0,len(w)):
            P1[i], P2[i] = rod_heterodimer_fromfile.analytic_twooscill(w=w[i])
        plt.plot(w,P1/max(P1), label='part 1, analytic')
        plt.plot(w,P2/max(P1), label='part 2, analytic')
        # plt.plot(w, (P1+P2), label='total, analytic')
        plt.legend()

    def numeric_power_abs(self, w):
        """Solves for the power absorbed by two coupled oscillators (q.s.) 
        
        Keywords: 
        w -- energy [eV]
        """
        k = w/hbar_eVs*np.sqrt(eps_b)/c
        w0, m, gamNR = self.dipole_parameters()

        val, vec, matrix = self.make_matrix(k=k)
        M = matrix + np.identity(self.mat_size)*(-w**2)
        K = e**2/m*self.drive
        alpha = np.matmul(np.linalg.inv(M), K)
        I0 = 10**12 # [erg s^-1 cm^-2] 10^9 W/m^2
        E0 = np.sqrt(I0*8*np.pi/c*np.sqrt(eps_b)) # [g^1/2 s^-1 cm^-1/2]
        Pow = 1/2*np.pi*w**2*E0**2/e**2*m*self.gamNR*np.abs(alpha)**2
        # power_absorbed = np.reshape(Pow, (1,self.mat_size))
        # abs_cross = power_absorbed / (c*E0**2/(8*np.pi))

        return abs_cross*10**8

    def plot_numeric_power_abs(self):
        w = np.arange(1,3.5,.005)
        Pow = np.zeros((len(w),self.mat_size))
        for i in range(0,len(w)):
            Pow[i,:] = self.numeric_power_abs(w=w[i])

        plt.plot(w,Pow[:,0], label=1)
        plt.plot(w,Pow[:,1], '--',label=2)
        plt.show()

    #     # for part in range(0, self.mat_size):
    #     #     plt.plot(w,Pow[:,part], label=part+1)
    #     # plt.plot(w, np.sum(Pow, axis=1),label='total')
    #     plt.legend()

    # def write_numeric_power_abs(self):
    #     ''' Writes power absorbed by each dipole to file. Each row cooresponds to each dipole.'''
    #     w = np.arange(1,3.5,.005)
    #     Pow = np.zeros((len(w), self.mat_size))
    #     for i in range(0,len(w)):
    #         Pow[i,:] = self.numeric_power_abs(w=w[i])
    #     # towrite = np.transpose(np.column_stack((w, Pow)))
    #     add_dip_info = np.column_stack((data[:,0:5], np.transpose(Pow)))
    #     w = np.append(np.array([0, 0, 0, 0, 0]), w)
    #     all_together = np.vstack((w,add_dip_info))
    #     np.savetxt(str('database/outputs/')+str(shapename)+str('.txt'), all_together, fmt="%2.5E")

    # def read_numeric_power_abs(self):
    #     file = str('database/outputs/')+str(shapename)+str('.txt')
    #     data = np.loadtxt(file)
    #     print(data[5,0:5])
    #     # plt.plot(data[0,5:], data[1,5:])
    #     # plt.show()

# shapename = str('4mer_50')
# filename = str('database/outputs/')+shapename+str('.txt')

# data = np.loadtxt(filename,skiprows=1)
# asym_trimer = CoupledOscillators(
#         3, # num particles
#         2, # number of dipoles per particle 
#         data[:,0:2], # particle centers [particle 1, particle 2, ...]
#         data[:,2:4], # unit vectors defining the orientation of each dipole
#         data[:,4], # radii or semi-axis corresponding to that dipole 
#         data[:,5], # kind of particle (currently only takes prolate spherioids and spheres)
#         '', # if prolate spherioid with only long axis dipole, fill in the semi-minor axis length here
#         data[:,6], # which dipoles are driven with plane wave light 
#         )
# asym_trimer.write_numeric_power_abs()
# asym_trimer.read_numeric_power_abs()

# plt.show()

##################################################################
################## Troubleshooting Oligomers ##################### 
##################################################################
data = np.loadtxt('dimer.txt',skiprows=1)
dimer = CoupledOscillators(
        2, # num particles
        1, # number of dipoles per particle 
        data[:,0:2], # particle centers [particle 1, particle 2, ...]
        data[:,2:4], # unit vectors defining the orientation of each dipole
        data[:,4], # radii or semi-axis corresponding to that dipole 
        data[:,5], # kind of particle (currently only takes prolate spherioids and spheres)
        30E-7, # if prolate spherioid with only long axis dipole, fill in the semi-minor axis length here
        data[:,6],
        )

# dimer.see_vectors()
# plt.show()
dimer.plot_numeric_power_abs()
# data_triangle = np.loadtxt('inputs_triangle.txt',skiprows=1)
# triangle = CoupledOscillators(
#         3, # num particles
#         1, # number of dipoles per particle 
#         data_triangle[:,0:2], # particle centers [particle 1, particle 2, ...]
#         data_triangle[:,2:4], # unit vectors defining the orientation of each dipole
#         data_triangle[:,4], # radii or semi-axis corresponding to that dipole 
#         data_triangle[:,5], # kind of particle (currently only takes prolate spherioids and spheres)
#         )

# data_4mer = np.loadtxt('database/inputs_rodhomodimer.txt',skiprows=1)
# pentamer = CoupledOscillators(
#         2, # num particles
#         1, # number of dipoles per particle 
#         data_4mer[:,0:2], # particle centers [particle 1, particle 2, ...]
#         data_4mer[:,2:4], # unit vectors defining the orientation of each dipole
#         data_4mer[:,4], # radii or semi-axis corresponding to that dipole 
#         data_4mer[:,5], # kind of particle (currently only takes prolate spherioids and spheres)
#         37E-7, # semi-minor axis of prolate sphereoid (included here because it's not defined in data file)
#         data[:,6], # which dipoles are driven with plane wave light 
#         )
# # pentamer.see_vectors()
# plt.show()

# x = data_4mer[:,0]*1E7
# y = data_4mer[:,1]*1E7

# plt.scatter(x, y)
# plt.scatter(x[1], y[1],color='k')
# plt.axis('equal')
# plt.show()


# data_pentamer = np.loadtxt('inputs_newpentamer.txt',skiprows=1)
# pentamer = CoupledOscillators(
#         5, # num particles
#         1, # number of dipoles per particle 
#         data_pentamer[:,0:2], # particle centers [particle 1, particle 2, ...]
#         data_pentamer[:,2:4], # unit vectors defining the orientation of each dipole
#         data_pentamer[:,4], # radii or semi-axis corresponding to that dipole 
#         data_pentamer[:,5], # kind of particle (currently only takes prolate spherioids and spheres)
#         37E-7, # semi-minor axis of prolate sphereoid (included here because it's not defined in data file)
#         )
# pentamer.see_vectors()
# plt.show()

# x = data_pentamer[:,0]*1E7
# y = data_pentamer[:,1]*1E7

# xcirc = np.array([x[0], x[1], x[2], x[3], x[4], x[0]])
# ycirc = np.array([y[0], y[1], y[2], y[3], y[4],y[0]])

# # x_unitvec = data_pentamer[:,2]
# # y_unitvec = data_pentamer[:,3]

# plt.scatter(xcirc, ycirc)
# plt.scatter(x[0], y[0],color='k')
# plt.axis('equal')
# plt.show()

# side1 = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
# side2 = np.sqrt((x[1]-x[2])**2 + (y[1]-y[2])**2)
# side3 = np.sqrt((x[2]-x[3])**2 + (y[2]-y[3])**2)
# side4 = np.sqrt((x[3]-x[4])**2 + (y[3]-y[4])**2)
# side5 = np.sqrt((x[4]-x[0])**2 + (y[4]-y[0])**2)

# print(side1, side2, side3, side4, side5)
# # print(np.arctan2(y_unitvec, x_unitvec)*180/np.pi)
# # print(side1)
# plt.axis('equal')
# plt.show()
# # print([x_unitvec[0], y_unitvec[0]])

# angle01 = np.dot([x_unitvec[0], y_unitvec[0]], [x_unitvec[1], y_unitvec[1]])
# angle12 = np.dot([x_unitvec[1], y_unitvec[1]], [x_unitvec[2], y_unitvec[2]])
# angle23 = np.dot([x_unitvec[2], y_unitvec[2]], [x_unitvec[3], y_unitvec[3]])
# angle34 = np.dot([x_unitvec[3], y_unitvec[3]], [x_unitvec[4], y_unitvec[4]])
# angle40 = np.dot([x_unitvec[4], y_unitvec[4]], [x_unitvec[0], y_unitvec[0]])

# print(angle01*180/np.pi)
# print(angle12*180/np.pi)
# print(angle23*180/np.pi)
# print(angle34*180/np.pi)
# print(angle40*180/np.pi)
