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

class CoupledOscillators:
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
            
    def dipole_parameters(self):
        """Sets the physical dipole parameters. This is assuming the dipoles represent spheres or prolate spheroids.
        """
        w0 = np.zeros(self.num_dip*self.num_part) # initiaize array for resonance frequency for each dipole
        m = np.zeros(self.num_dip*self.num_part) # initiaize array for effective for each dipole
        gamNR = np.zeros(self.num_dip*self.num_part) # initiaize array for nonradiative damping for each dipole
        # The following three parameters are fits to gold JC data. 
        # See the included script, "fit_jc_to_drude.py" to learn more.
        wp = 9. # [eV], bulk plasma frequency 
        eps_inf = 9.7 # [unitless], static dielectric response of ionic background 
        gamNR_qs = 0.073 # [eV]
        for row in range(0,self.num_dip*self.num_part):
            if self.kind[row] == 0: # sphere
                D = 1.
                li = self.radii[row]
                v = 4./3*np.pi*(self.radii[row])**3 # [cm^3], volume of sphere
                Li = 1./3
                w0_qs = np.sqrt(wp**2/(eps_inf+2*eps_b)) # [eV], quasi-static plasmon resonance frequency 
                m_qs = 4.*np.pi*e**2/(v*(w0_qs/hbar_eVs)**2)*(eps_inf + 2*eps_b)/(3**2*eps_b)

                # Long wavelength approximation (https://www.osapublishing.org/josab/viewmedia.cfm?uri=josab-26-3-517&seq=0)
                m[row] =  m_qs# + D*e**2/(li*c**2) # [g], mass with radiation damping
                w0[row] = w0_qs#*np.sqrt(m_qs/m[row]) # [eV], plasmon resonance frequency with radiation damping
                gamNR[row] = gamNR_qs# * m_qs/m[row] # [eV], adjusted nonradiative damping 

            # if self.kind[row] == 1: # prolate spheroid
            #     cs = self.radii[row] # [cm], semi-major axis of prolate sphereiod 
            #     # find which row corresponds to the semi-minor axis of prolate spheroid (this is required to calculate dipole parameters)
            #     idx = np.where( (self.centers[:,0] == self.centers[row,0]) &  # the semi-minor axis will have the same origin
            #             (self.centers[:,1] == self.centers[row,1]) & # the semi-minor axis will have the same origin
            #             (self.kind == 2) # the semi-minor axis will have kind = 2
            #             )
            #     if self.optional_semiaxis == '': 
            #         a = self.radii[idx] # [cm], semi-minor axis of prolate spheriod 
            #     if self.optional_semiaxis != '': 
            #         a = self.optional_semiaxis # [cm], semi-minor axis of prolate spheriod 
            #     es = (cs**2 - a**2)/cs**2 # [unitless]
            #     Lz = (1-es**2)/es**3*(-es+1./2*np.log((1+es)/(1-es))) # [unitless]
            #     Ly = (1-Lz)/2 # [unitless]  
            #     D = 3./4*((1+es**2)/(1-es**2)*Lz + 1) # [unitless]
            #     V = 4./3*np.pi*a**2*cs # [cm^3]
            #     # for semi-major axis
            #     li = cs; Li = Lz 
            #     w0_qs = np.sqrt(wp**2/((eps_inf-1)+1/Li)) # [eV], quasi-static plasmon resonance frequency 

            #     m_qs= 4*np.pi*e**2*((eps_inf-1)+1/Li)/((w0_qs/hbar_eVs)**2*V/Li**2) # [g] 
            #     m[row] = m_qs + D*e**2/(li*c**2) # [g]
            #     w0[row] = (w0_qs)*np.sqrt(m_qs/m[row]) # [eV]
            #     gamNR[row] = 0.07*(m_qs/m[row]) # [eV]
            #     # for semi-minor axis 
            #     if self.optional_semiaxis == '':
            #         li = a
            #         Li = Ly
            #         m_qs= 4*np.pi*e**2*((eps_inf-1)+1/Li)/((w0_qs/hbar_eVs)**2*V/Li**2) # g 
            #         m[idx] = m_qs + D*e**2/(li*c**2) # g (charge and speed of light)
            #         w0[idx] = (w0_qs)*np.sqrt(m_qs/m[idx]) # 1/s
            #         gamNR[idx] = 0.07*(m_qs/m[idx]) 
        return w0, m, gamNR # [eV], [g], [eV]
    
    def coupling(self, dip_i, dip_j, k): 
        """Calculates the off diagonal matrix elements, which is the 
        coupling between the ith and jth dipole divided by the effective 
        mass of the ith dipole.
        
        Keyword arguments:
        dip_i -- ith dipole
        dip_j -- jth dipole
        k -- wave vector [1/cm]
        """
        k = np.real(k) # [1/cm]
        r_ij = self.centers[dip_i,:] - self.centers[dip_j,:] # [cm], distance between ith and jth dipole 
        mag_rij = np.linalg.norm(r_ij) 
        # print(1/mag_rij, k)
        if mag_rij == 0: 
        # If i and j are at the same location, the coupling (g) is zero (prevents a divide by zero error.)
            g=0
        else:
            nhat_ij = r_ij / mag_rij # [unitless], unit vector in r_ij direction.
            xi_hat = self.unit_vecs[dip_i] # [unitless], unit vector of the ith dipole
            xj_hat = self.unit_vecs[dip_j] # [unitless], unit vector of the jth dipole
            xi_dot_nn_dot_xj = np.dot(xi_hat, nhat_ij)*np.dot(nhat_ij, xj_hat) # [unitless]
            nearField = ( 3.*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat) ) / mag_rij**3 # [1/cm^3], near field coupling.
            intermedField = 1j*k*(3*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij**2 # [1/cm^3], intermediate field coupling
            farField = k**2*(xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij # [1/cm^3], far field coupling
            g = e**2 * hbar_eVs**2 / eps_b * ( nearField  - intermedField - farField ) * np.exp(1j*k*mag_rij) # [g eV^2], total radiative coupling
            # print('%.2E %.2E' % (nearField, np.imag(intermedField)))
            # print(k*c/np.sqrt(eps_b)*hbar_eVs)
            # print(k*mag_rij)
            # print(np.exp(1j*k*mag_rij))
            # print(g)
        return g # [g eV^2]
    
    def make_matrix(self, k):
        """Forms the matrix A for A(w)*x = w^2*x. 
        
        Keywords: 
        k -- wave vector [1/cm]
        """
        matrix = np.zeros( (self.mat_size, self.mat_size) ,dtype=complex) 
        w_guess = k*c/np.sqrt(eps_b)*hbar_eVs # [eV], left-hand size w 
        gam = self.gamNR + (np.real(w_guess))**2*(2.0*e**2)/(3.0*self.m*c**3)/hbar_eVs # [eV], radiative damping for dipole i
        matrix[( np.arange(self.mat_size), np.arange(self.mat_size) )] = self.w0**2 - 1j*gam*w_guess # [eV^2], on-diagonal matrix elements
        for dip_i in range(0 , self.mat_size): 
            for dip_j in range(0, self.mat_size): 
                    if dip_i != dip_j:
                        matrix[ dip_i, dip_j] = -self.coupling(dip_i=dip_i, dip_j=dip_j, k=k)/self.m[dip_i] # [eV^2], off-diagonal matrix elements
        eigval, eigvec = np.linalg.eig(matrix)
        return eigval, eigvec, matrix #[eV^2], [unitless], [eV^2]

    def iterate(self):
        """Solves A(w)*x = w^2*x for w and x. This is done by guessing the first left-side w, calculating
        eigenvalues (w^2) and eigenvectors (x) of A(w), then comparing the initial guess w to the calculated w.
        If the difference of those values is less than the variable precision, use the new w and continue the 
        cycle. The code will stop (converge), when the both w agree as well as the eigenvectors, x. Note that if
        the while loop surpasses 100 trials for a single eigenvalue/eigenvector set, the selection of the left-hand
        side w is modified to take into account previous trials. 
        """
        final_eigvals = np.zeros(self.mat_size,dtype=complex) 
        final_eigvecs = np.zeros( (self.mat_size, self.mat_size), dtype=complex) 
        w_init = -1j*self.gamNR/2. + np.sqrt(-self.gamNR**2/4.+self.w0**2) # [eV], Initial guess of left-hand size w. This is the result for an isolated nonradiative dipole.
        for mode in range(0,self.mat_size): # Converge each mode individually.
            eigval_hist = np.array([w_init[mode], w_init[mode]*1.1],dtype=complex) 
            eigvec_hist = np.zeros((self.mat_size, 2))
            count = 0
            while (np.abs((np.real(eigval_hist[0]) - np.real(eigval_hist[1])))  > 10**(-prec)) or \
                  (np.abs((np.imag(eigval_hist[0]) - np.imag(eigval_hist[1]))) > 10**(-prec)) or \
                  (np.sum(np.abs((eigvec_hist[:,0] - eigvec_hist[:,1]))) > 10**(-prec)):
                w_guess = eigval_hist[0]
                if count > 1000: 
                    denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
                    w_guess = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom 
                k = w_guess/hbar_eVs*np.sqrt(eps_b)/c # [1/cm]
                val, vec, H = self.make_matrix(k=k)
                # print(val)
                amp = np.sqrt(np.abs(val))
                phi = np.arctan2(np.imag(val), np.real(val))
                energy = amp*(np.cos(phi/2.)+1j*np.sin(phi/2.))
                post_sort_val = energy[energy.argsort()] # sort the evals and evecs so that we are comparing the same eval/evec as the previous trial.
                post_sort_vec = vec[:,energy.argsort()]
                this_val = post_sort_val[mode] # [eV]
                this_vec = post_sort_vec[:,mode] # [unitless]
                eigval_hist = np.append(this_val, eigval_hist)
                eigvec_hist = np.column_stack((this_vec, eigvec_hist))
                count = count + 1      
                # print(mode, count, np.real(this_val), np.real(this_vec))
            final_eigvals[mode] = eigval_hist[0]
            final_eigvecs[:,mode] = eigvec_hist[:,0]
        return final_eigvals, final_eigvecs  
    
    def see_vectors(self):
        """Plot the convereged eigenvectors."""
        final_eigvals, final_eigvecs = self.iterate()
        dip_ycoords = self.centers[:,0]
        dip_zcoords = self.centers[:,1]  
        plt.figure(1, figsize=[5,3])
        for mode in range(0,self.mat_size):
            w = final_eigvals[mode] # [eV]
            v = np.real(final_eigvecs[:,mode]) # [unitless]
            plt.subplot(1,self.mat_size,mode+1)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.title('%.2f' % (np.real(w)) + ' + i%.2f eV' % (np.imag(w)), fontsize=10)
            p = v[...,np.newaxis]*self.unit_vecs
            if self.num_dip == 1: p_perpart = p
            else: p_perpart = p[:int(self.mat_size/self.num_dip),:] + p[int(self.mat_size/self.num_dip):,:] 
            ymin = min(dip_ycoords)-100E-7; ymax = max(dip_ycoords)+100E-7
            zmin = min(dip_zcoords)-100E-7; zmax = max(dip_zcoords)+100E-7
            plt.quiver(dip_ycoords[:int(self.mat_size/self.num_dip)], dip_zcoords[:int(self.mat_size/self.num_dip)], p_perpart[:,0], p_perpart[:,1], pivot='mid', 
                width=.5, #shaft width in arrow units 
                scale=1., 
                headlength=5,
                headwidth=5.,
                minshaft=4., 
                minlength=.01
                )
            plt.xlim([ymin, ymax])
            plt.ylim([zmin, zmax])
            # plt.axis('equal')
            plt.yticks([]); plt.xticks([])
            plt.scatter(dip_ycoords, dip_zcoords,c='blue',s=20)

    # def analytic_twooscill(self,w):
    #     """Solves for the power absorbed by two coupled oscillators (q.s.) 
        
    #     Keywords: 
    #     w -- energy [eV]
    #     """
    #     I0 = 10**12 # [erg s^-1 cm^-2] 10^9 W/m^2
    #     E0 = np.sqrt(I0*8*np.pi/c*np.sqrt(eps_b)) # [g^1/2 s^-1 cm^-1/2]
    #     dip_i = 0; dip_j = 1
    #     k = w/hbar_eVs*np.sqrt(eps_b)/c
    #     w0, m, gamNR = self.dipole_parameters()
    #     m1 = m[0]; m2 = m[1]; 
    #     w01 = w0[0]; w02 = w0[1]
    #     g = np.real(self.coupling(dip_i=dip_i, dip_j=dip_j, k=k)) # off-diagonal matrix elements
    #     gam = self.gamNR + (w)**2*(2.0*e**2)/(3.0*self.m*c**3)/hbar_eVs # radiative damping for dipole i

    #     Omega1 = -w**2 - 1j*w*gam[0] + w01**2
    #     Omega2 = -w**2 - 1j*w*gam[1] + w02**2
    #     alph1 = np.real(Omega1); alph2 = np.real(Omega2)
    #     beta1 = np.imag(Omega1); beta2 = np.imag(Omega2)

    #     a1 = g*m1*m2*(alph1*alph2 - beta1*beta2) + m1*m2**2*alph1*(alph2**2 + beta2**2) - g**2*m2*alph2 - g**3 # [g^3 eV^6]
    #     b1 = g*m1*m2*(alph2*beta1 + alph1*beta2) + m1*m2**2*beta1*(alph2**2 + beta2**2) + g**2*m2*beta2 # [g^3 eV^6]
    #     P1 = 1./2*m1*self.gamNR[0]*(e*E0)**2*w**2/(np.abs(m1*m2*Omega1*Omega2 - g**2)**4)*(a1**2 + b1**2)*hbar_eVs # [g cm^2 s^-3]

    #     a2 = g*m1*m2*(alph1*alph2 - beta1*beta2) + m2*m1**2*alph2*(alph1**2 + beta1**2) - g**2*m1*alph1 - g**3 # [g^3 eV^6]
    #     b2 = g*m1*m2*(alph2*beta1 + alph1*beta2) + m2*m1**2*beta2*(alph1**2 + beta1**2) + g**2*m1*beta1 # [g^3 eV^6]
    #     P2 = 1./2*m2*self.gamNR[1]*(e*E0)**2*w**2/(np.abs(m1*m2*Omega1*Omega2 - g**2)**4)*(a2**2 + b2**2)*hbar_eVs # [g cm^2 s^-3]
    #     return P1, P2

    # def plot_analytic_twooscill(self):
    #     w = np.arange(1,4,.005)
    #     P1 = np.zeros(len(w))
    #     P2 = np.zeros(len(w))

    #     for i in range(0,len(w)):
    #         P1[i], P2[i] = rod_heterodimer_fromfile.analytic_twooscill(w=w[i])

    #     plt.plot(w,P1/max(P1), label='part 1, analytic')
    #     plt.plot(w,P2/max(P1), label='part 2, analytic')
    #     # plt.plot(w, (P1+P2), label='total, analytic')
    #     plt.legend()

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


