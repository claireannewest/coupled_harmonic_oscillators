import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn

n = 1
eps_b = n**2
e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]
wp = 8.959/hbar_eVs # [1/s], bulk plasma frequency 

eps_inf = 9.695 # [unitless], static dielectric response of ionic background 
gamNR_qs = 0.073/hbar_eVs # [1/s]

class DipoleParameters:
    def __init__(self, centers, orient, all_radii):
        """Defines the different system parameters.
        
        Keyword arguments:
        centers -- coordinates of centers of each prolate spheorid [cm]
        orient -- orientation of the long axis of prolate spheroid 
        all_radii -- radii of the oscillators [cm]
        """
        self.centers = centers
        self.orient = orient
        self.all_radii = all_radii

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

    def mie_coefficents(self, w,radius):
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

    def alpha(self, w, radius, kind): 
        ### assumes z axis is long axis, and y = x 
        k = w/c
        # cs equals long axis, a equals short axis
        cs = radius[0]; a = radius[1]
        if cs == a:
            ### it's a sphere
            a, b = mie_coefficents(w,radius[0])
            alpha = 3/(2.*k**3)*1j*(a)#+b) 
        else:
            ### it's a spheroid 
            V = 4/3*np.pi*cs*a**2
            eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
            es = np.sqrt(1 - a**2/cs**2)
            Lz = (1-es**2)/es**2*(-1 + 1/(2*es)*np.log((1+es)/(1-es)))
            Dz = 3./4*((1+es**2)/(1-es**2)*Lz + 1) # [unitless]
            Ly = (1 - Lz )/2
            Dy = a/(2*cs)*(3/es*np.tanh(es)**(-1)-Dz) 
            if kind == 'prolate_alonglong':
                lE = cs
                L = Lz
                D = Dz
            if kind == 'prolate_alongshort':
                lE = a
                L = Ly
                D = Dy
            alphaR = V/(4*np.pi)*(eps-1)/(1+L*(eps-1))
            alphaMW = alphaR / (1 - k**2/lE *D*alphaR - 1j*2*k**3/3*alphaR)
            alpha = alphaMW
        return alpha

class CalculateCrossSections:
    def __init__(self, centers, orient, all_radii):
        """Defines the different system parameters.
        
        Keyword arguments:
        centers -- coordinates of centers of each prolate spheorid [cm]
        orient -- orientation of the long axis of prolate spheroid 
        all_radii -- radii of the oscillators [cm]
        """
        self.centers = centers
        self.orient = orient
        self.all_radii = all_radii

    def A_ij(self, dip_i, dip_j, k):
        ''' off diagonal block terms in A_tilde '''
        A_ij = np.zeros( (3, 3) ,dtype=complex) 
        r_ij = self.centers[dip_i,:] - self.centers[dip_j,:] # [cm], distance between ith and jth dipole 
        r = np.sqrt(r_ij[0]**2+r_ij[1]**2+r_ij[2]**2)
        rx = r_ij[0]; ry = r_ij[1]; rz = r_ij[2]
        A_ij[0,0] = np.exp(1j*k*r)/r**3*(k**2*(rx*rx-r**2) + (1-1j*k*r)/r**2*(-3*rx*rx+r**2))
        A_ij[0,1] = np.exp(1j*k*r)/r**3*(k**2*(rx*ry) + (1-1j*k*r)/r**2*(-3*rx*ry))
        A_ij[0,2] = np.exp(1j*k*r)/r**3*(k**2*(rx*rz) + (1-1j*k*r)/r**2*(-3*rx*rz))
        A_ij[1,0] = A_ij[0,1]
        A_ij[1,1] = np.exp(1j*k*r)/r**3*(k**2*(ry*ry-r**2) + (1-1j*k*r)/r**2*(-3*ry*ry+r**2))
        A_ij[1,2] = np.exp(1j*k*r)/r**3*(k**2*(ry*rz) + (1-1j*k*r)/r**2*(-3*ry*rz))
        A_ij[2,0] = A_ij[0,2]
        A_ij[2,1] = A_ij[1,2]
        A_ij[2,2] = np.exp(1j*k*r)/r**3*(k**2*(rz*rz-r**2) + (1-1j*k*r)/r**2*(-3*rz*rz+r**2))
        return A_ij


    def A_ii(self, dip_i, dip_j, k):
        '''On diagonal block terms in A_tilde. Requires aligned along z direction.'''
        w = k*c
        A_ii = np.zeros( (3, 3) ,dtype=complex) 
        dip_params = DipoleParameters(self.centers, self.orient, self.all_radii)
        A_ii[0,0] = dip_params.alpha(w=w, radius=self.all_radii[dip_i, : ], kind='prolate_alongshort')**(-1)
        A_ii[1,1] = dip_params.alpha(w=w, radius=self.all_radii[dip_i, : ], kind='prolate_alongshort')**(-1)
        A_ii[2,2] = dip_params.alpha(w=w, radius=self.all_radii[dip_i, : ], kind='prolate_alonglong')**(-1)
        return A_ii

    def A_tilde(self, w):
        '''A_tilde = [3*N, 3*N]'''
        k = w/c
        A_tilde = np.zeros( (3*N, 3*N) ,dtype=complex) 
        for i in range(0 , N): 
            for j in range(0, N):
                if i == j:  
                    A_tilde[3*i : 3*(i+1), 3*i : 3*(i+1)] = self.A_ii(dip_i=i, dip_j=i, k=k)
                if i != j:
                    A_tilde[3*i : 3*(i+1), 3*j : 3*(j+1)] = self.A_ij(dip_i=i, dip_j=j, k=k)
        return A_tilde

    def P_tilde(self, w, drive):
        E0_tilde = np.zeros((3*N, 1))
        P_tilde = np.zeros((3*N, 1),dtype=complex)
        for i in range(0, N):
            E0_tilde[3*i,:] = drive[0]
            E0_tilde[3*i+1,:] = drive[1]
            E0_tilde[3*i+2,:] = drive[2]
        P_tilde = np.linalg.solve(self.A_tilde(w), E0_tilde)
        return P_tilde

    def cross_sects(self, w, drive):
        ''' Works for spheres up to 50 nm radii, and in the window < 3.0 eV '''
        k = w/c
        P = self.P_tilde(w=w, drive=drive)
        P_each = np.zeros((3, N),dtype=complex)
        E_each = np.zeros((3, N),dtype=complex)
        Cext_each = np.zeros(N)
        Csca_each = np.zeros(N)

        for i in range(0, N):
            # Evaluate the cross sections of each particle separately
            dip_params = DipoleParameters(self.centers, self.orient, self.all_radii)
            P_each[:,i] = P[3*i:3*(i+1), 0]
            E_each[0,i] = dip_params.alpha(w=w, radius=self.all_radii[i, :],kind='prolate_alongshort')**(-1)*P_each[0,i]
            E_each[1,i] = dip_params.alpha(w=w, radius=self.all_radii[i, :],kind='prolate_alongshort')**(-1)*P_each[1,i]
            E_each[2,i] = dip_params.alpha(w=w, radius=self.all_radii[i, :],kind='prolate_alonglong')**(-1)*P_each[2,i]

            Cext_each[i] = 4*np.pi*k*np.imag( np.sum( P_each[:,i]*np.conj(E_each[:,i])) ) *10**8
            Csca_each[i] = 4*np.pi*k*2/3*k**3*np.real( np.sum(P_each[:,i]*np.conj(P_each[:,i])) ) *10**8  
        Cabs_each = Cext_each - Csca_each
        return Cabs_each

    def plot_analytics(self, drive):
        w = np.arange(.5,3,.005)/hbar_eVs
        P = np.zeros((N, len(w)))

        for i in range(0,len(w)):
            P[:,i] = self.cross_sects(w=w[i], drive=drive)

        for i in range(0, N):
            plt.plot(w*hbar_eVs, P[i,:], label=str('spheroid ')+str(i))
        plt.plot(w*hbar_eVs, np.sum(P, axis=0), label='total cross sect.')

        plt.xlabel('Energy [eV]')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.ylabel('Abs. Cross-section [$\mu$m$^2$]')
        plt.tight_layout()
        plt.legend(frameon=False,loc='upper right')
        # plt.show()

    def write_datafiles(self,filename):
        ''' Writes power absorbed by each particle and each laser polarization
        '''
        w = np.arange(0.5,3.0,.01)/hbar_eVs
        write_file = np.zeros((1+3*N, len(w)+8))
        P_orientx = np.zeros((N, len(w)))
        P_orienty = np.zeros((N, len(w)))
        P_orientz = np.zeros((N, len(w)))

        for i in range(0,len(w)):
            P_orientx[:,i] = self.cross_sects(w=w[i], drive=np.array([1, 0, 0]))
            P_orienty[:,i] = self.cross_sects(w=w[i], drive=np.array([0, 1, 0]))
            P_orientz[:,i] = self.cross_sects(w=w[i], drive=np.array([0, 0, 1]))
        write_file[0,8:] = w*hbar_eVs
        for i in range(0, N):
            write_file[3*i+1, 0:3] = self.centers[i, :]
            write_file[3*i+2, 0:3] = self.centers[i, :]
            write_file[3*i+3, 0:3] = self.centers[i, :]
        
            write_file[3*i+1, 3:6] = self.orient[i,:]
            write_file[3*i+2, 3:6] = self.orient[i,:]
            write_file[3*i+3, 3:6] = self.orient[i,:]

            write_file[3*i+1, 6:8] = self.all_radii[i, :]
            write_file[3*i+2, 6:8] = self.all_radii[i, :]
            write_file[3*i+3, 6:8] = self.all_radii[i, :]

            write_file[3*i+1, 8:] = P_orientx[i,:]
            write_file[3*i+2, 8:] = P_orienty[i,:]
            write_file[3*i+3, 8:] = P_orientz[i,:]
        np.savetxt(str('datafiles_for_learning/')+str(filename)+str('.txt'), write_file, fmt="%2.5E")

    def plot_twospectra(self, filename):
        data = np.loadtxt(filename,skiprows=1)
        energy = data[:,0]
        ext_cross = data[:,1]
        abs_cross = data[:,2]
        sca_cross = data[:,3]
        plt.plot(energy, abs_cross, 'r',alpha=.5,label='simulation')

    def plot_fromfile(self, drive):
        zrad1, yrad1 = self.all_radii[0, :]
        zrad2, yrad2 = self.all_radii[1, :]
        ylength = int(round(yrad1*1E7))
        zlength = int(round(zrad1*1E7))
        zgaplen = int(round((self.centers[1,2] - self.centers[0,2] - zrad1 - zrad2)*1E7))

        name = str('twoprolates_homo_')+str(ylength)+str('_')+str(zlength)\
                                           +str('_z')+str(zgaplen)

        data = np.loadtxt(str('datafiles_for_learning/')+str(name)+str('.txt'))


        if drive[0] == 1:
            plt.plot(data[0,8:], data[1,8:],'--',label='fromfile')
            plt.plot(data[0,8:], data[4,8:],'--',label='fromfile')
        if drive[1] == 1:
            plt.plot(data[0,8:], data[2,8:],'--',label='fromfile')
            plt.plot(data[0,8:], data[5,8:],'--',label='fromfile')
            plt.plot(data[0,8:], data[2,8:]+data[5,8:],'--',label='model')

        if drive[2] == 1:
            plt.plot(data[0,8:], data[3,8:],'--',label='fromfile')
            plt.plot(data[0,8:], data[6,8:],'--',label='fromfile')
            plt.plot(data[0,8:], data[3,8:]+data[6,8:],label='model')

def write_em_all():
    for zlength in range(100, 110, 10):
        for zgaplen in range(10, 50, 10):
            ylength=80
            ygaplen = 0

            yrad1 = ylength*1.E-7; zrad1 = zlength*1.E-7
            yrad2 = ylength*1.E-7; zrad2 = zlength*1.E-7

            gapz = zgaplen*1.E-7
            gapy = ygaplen*1.E-7
            N = 2 # number of particles

            def_centers = np.zeros((N, 3))
            def_orient = np.zeros((N, 3))
            def_all_radii = np.zeros((N, 2))

            if ygaplen == 0:
                def_centers[0,:] = np.array([ 0, 0, -zrad1-gapz/2 ])
                def_centers[1,:] = np.array([ 0,  0, zrad2+gapz/2 ])

            if zgaplen == 0:
                def_centers[0,:] = np.array([ 0, -yrad1-gapy/2, 0 ])
                def_centers[1,:] = np.array([ 0,  yrad2-gapy/2, 0 ])

            if ygaplen != 0 and zgaplen != 0:
                def_centers[0,:] = np.array([ 0, -yrad1-gapy/2, -zrad1-gapz/2 ])
                def_centers[1,:] = np.array([ 0,  yrad2-gapy/2, zrad2+gapz/2 ])

            def_all_radii[0, :] = np.array([zrad1, yrad1])
            def_all_radii[1, :] = np.array([zrad2, yrad2])

            calc_abs = CalculateCrossSections(
                    def_centers, # particle centers 
                    def_orient, # orientation of long axis prolate spheroid 
                    def_all_radii, # semi-major and semi-minor axis lengths of each spheroid
                    )

            name = str('twoprolates_homo_')+str(ylength)+str('_')+str(zlength)\
                                           +str('_z')+str(zgaplen)
            calc_abs.write_datafiles(filename=name)

# write_em_all()

zlength = 100
ylength=80

zgaplen = 30
ygaplen = 0

yrad1 = ylength*1.E-7; zrad1 = zlength*1.E-7
yrad2 = ylength*1.E-7; zrad2 = zlength*1.E-7

gapz = zgaplen*1.E-7
gapy = ygaplen*1.E-7
N = 2 # number of particles

def_centers = np.zeros((N, 3))
def_orient = np.zeros((N, 3))
def_all_radii = np.zeros((N, 2))

def_centers[0,:] = np.array([ 0, 0, -zrad1-gapz/2 ])
def_centers[1,:] = np.array([ 0,  0, zrad2+gapz/2 ])

def_orient[0, :] = np.array([0, 0, 1])
def_orient[1, :] = np.array([0, 0, 1])

def_all_radii[0, :] = np.array([zrad1, yrad1])
def_all_radii[1, :] = np.array([zrad2, yrad2])


test = CalculateCrossSections(def_centers, def_orient, def_all_radii)
test.plot_fromfile(drive=np.array([0,0,1]))
test.plot_twospectra(filename='check_with_simulation/simulated_spectra/two_ellipsoids/Spectrum_bemret_homo_80_100_z30_polz')
# plt.xlim([2,3.])
plt.legend()
plt.show()


# class NormalModes:
#     def __init__(self, centers, orient, all_radii):
#         """Defines the different system parameters.
        
#         Keyword arguments:
#         centers -- coordinates of centers of each particle
#         orient -- orientation of the long axis of prolate spheroid 
#         centers -- particle centers [cm]
#         orientaitons -- unit vectors defining each dipole [unitless]
#         radii -- radii of the oscillators [cm]
#         kind -- which kind of dipole (0 = sphere, 1 = long axis prolate spheroid, 2 = short axis prolate spheroid)
#         optional_semiaxis -- define the semi-axis of prolate sphereiod if it isn't defined in data file
#         """
#         self.centers = centers
#         self.orient = orient
#         self.all_radii = all_radii

#     def alphaprime_inv(self, w, radius, kind):        
#         k = w/c
#         # cs equals long axis, a equals short axis
#         cs = radius[0]; a = radius[1]
#         if cs == a:
#             ### it's a sphere
#             a, b = mie_coefficents(w,radius[0])
#             alpha = 3/(2.*k**3)*1j*(a)#+b) 
#         else:
#             ### it's a spheroid 
#             V = 4/3*np.pi*cs*a**2
#             eps = eps_inf - wp**2/(w**2 + 1j*w*gamNR_qs)
#             es = np.sqrt(1 - a**2/cs**2)
#             Lz = (1-es**2)/es**2*(-1 + 1/(2*es)*np.log((1+es)/(1-es)))
#             Dz = 3./4*((1+es**2)/(1-es**2)*Lz + 1) # [unitless]
#             Ly = (1 - Lz )/2
#             Dy = a/(2*cs)*(3/es*np.tanh(es)**(-1)-Dz) 
#             if kind == 'prolate_alonglong':
#                 lE = cs
#                 L = Lz
#                 D = Dz
#             if kind == 'prolate_alongshort':
#                 lE = a
#                 L = Ly
#                 D = Dy
#             alphaR = V/(4*np.pi)*(eps-1)/(1+L*(eps-1))
#             alphaprime_inv = (1 - k**2/lE *D*alphaR - 1j*2/3*k**3*alphaR)/alphaR + w*hbar_eVs
#         return alphaprime_inv

#     def Aprime_ii(self, dip_i, dip_j, k):
#         '''On diagonal block terms in A_tilde. Requires aligned along z direction.'''
#         w = k*c
#         Aprime_ii = np.zeros( (3, 3) ,dtype=complex) 
#         Aprime_ii[0,0] = self.alphaprime_inv(w=w, radius=self.all_radii[dip_i, : ], kind='prolate_alongshort')
#         Aprime_ii[1,1] = self.alphaprime_inv(w=w, radius=self.all_radii[dip_i, : ], kind='prolate_alongshort')
#         Aprime_ii[2,2] = self.alphaprime_inv(w=w, radius=self.all_radii[dip_i, : ], kind='prolate_alonglong')
#         return Aprime_ii

#     def make_matrix(self, k):
#         """ Forms the matrix A for A_tilde'(k)*p = k*p """
#         #A_tilde = [3*N, 3*N]
#         Aprime_tilde = np.zeros( (3*N, 3*N) ,dtype=complex) 
#         for i in range(0 , N): 
#             for j in range(0, N):
#                 if i == j:  
#                     Aprime_tilde[3*i : 3*(i+1), 3*i : 3*(i+1)] = self.Aprime_ii(dip_i=i, dip_j=i, k=k)
#                 if i != j:
#                     calc_cross = CalculateCrossSections(self.centers, self.orient, self.all_radii)
#                     Aprime_tilde[3*i : 3*(i+1), 3*j : 3*(j+1)] = calc_cross.A_ij(dip_i=i, dip_j=j, k=k)
#         return Aprime_tilde

#     def find_eigen(self, w):
#         k = w/hbar_eVs/c
#         Aprime_tilde = self.make_matrix(k=k)
#         eigval, eigvec = np.linalg.eig(Aprime_tilde)
#         return eigval, eigvec, Aprime_tilde 

#     def iterate(self,prec=1):
#         """Solves Aprime_tilde(k)*p = k*p for w and x. This is done by guessing the first left-side k, calculating
#         eigenvalues k and eigenvectors (p) of Aprime_tilde, then comparing the initial guess k to the calculated k.
#         If the difference of those values is less than the variable precision, use the new k and continue the 
#         cycle. The code will stop (converge), when the both k's agree and all the eigenvectors, p agree. Note that if
#         the while loop surpasses 1000 trials for a single eigenvalue/eigenvector set, the selection of the left-hand
#         side w is modified to take into account previous trials. 
#         """
#         final_eigvals = np.zeros(3*N,dtype=complex) 
#         final_eigvecs = np.zeros( (3*N,3*N), dtype=complex) 
#         w_init = 2.5#/hbar_eVs#-1j*gamNR_qs/2. + np.sqrt(-gamNR_qs**2/4.+(2.5/hbar_eVs)**2) # [eV], Initial guess of left-hand size w. This is the result for an isolated nonradiative dipole.
#        # k_init = w_init/c
#         for mode in range(0,3*N): # Converge each mode individually.
#             eigval_hist = np.array([w_init , w_init*1.1],dtype=complex) 
#             eigvec_hist = np.zeros((3*N, 2))
#             count = 0
#             while (np.abs((np.real(eigval_hist[0]) - np.real(eigval_hist[1])))  > 10**(-prec)):# or \
#                               # (np.sum(np.abs((eigvec_hist[:,0] - eigvec_hist[:,1]))) > 10**(-prec)):

#                   # (np.abs((np.imag(eigval_hist[0]) - np.imag(eigval_hist[1]))) > 10**(-prec)) or \
#                   # (np.sum(np.abs((eigvec_hist[:,0] - eigvec_hist[:,1]))) > 10**(-prec)):
#                 w_guess = eigval_hist[0]
#                 if count > 100: 
#                     denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
#                     w_guess = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom 

#                 val, vec, Aprime_tilde = self.find_eigen(w=w_guess)

#                 amp = np.sqrt(np.abs(val))
#                 phi = np.arctan2(np.imag(val), np.real(val))
#                 val_re = amp*(np.cos(phi/2.)+1j*np.sin(phi/2.))
#                 post_sort_val = val_re[val_re.argsort()] # sort the evals and evecs so that we are comparing the same eval/evec as the previous trial.
#                 post_sort_vec = vec[:,val_re.argsort()]
#                 this_val = post_sort_val[mode] # [eV]
#                 this_vec = post_sort_vec[:,mode] # [unitless]
#                 eigval_hist = np.append(this_val, eigval_hist)
#                 eigvec_hist = np.column_stack((this_vec, eigvec_hist))
#                 count = count + 1      
#                 print(mode, count, this_val)#, np.real(this_vec))
#             final_eigvals[mode] = eigval_hist[0]
#             final_eigvecs[:,mode] = eigvec_hist[:,0]
#         return final_eigvals, final_eigvecs  
    
#     def see_vectors(self):
#         """Plot the convereged eigenvectors."""
#         final_eigvals, final_eigvecs = self.iterate()
#         dip_ycoords = self.centers[:,1]
#         dip_zcoords = self.centers[:,2]  
#         plt.figure(1, figsize=[5,3])
#         p_each = np.zeros((3, N))

#         for mode in range(0,3*N):
#             val = final_eigvals[mode] 

#             k = (3*2/(1j)*val)**(1/3)
#             w = k*c*hbar_eVs

#             vec = np.real(final_eigvecs[:,mode]) # [unitless]
#             plt.subplot(1,3*N,mode+1)
#             ax = plt.gca()
#             ax.set_aspect('equal', adjustable='box')
#             for particle in range(0, N):
#                 p_each[:,particle] = vec[3*particle:3*(particle+1)]

#             p0 = p_each[:,0]
#             p1 = p_each[:,1]

#             plt.quiver(dip_ycoords[1], dip_zcoords[1], p1[1], p1[2], pivot='mid', 
#                 width=.5, scale=1., headlength=5, headwidth=5., minshaft=4., minlength=.01)

#             plt.quiver(dip_ycoords[0], dip_zcoords[0], p0[1], p0[2], pivot='mid', 
#                 width=.5, scale=1., headlength=5, headwidth=5., minshaft=4., minlength=.01)
#             plt.title('%.2f' % (np.real(w)) , fontsize=10)

#             # plt.title('%.2f' % (np.real(w)) + ' + i%.2f eV' % (np.imag(w)), fontsize=10)
#         #     p = v[...,np.newaxis]*self.orient
#         #     p_perpart = p[:N,:] + p[N:,:] 
#         #     ymin = min(dip_ycoords)-100E-7; ymax = max(dip_ycoords)+100E-7
#         #     zmin = min(dip_zcoords)-100E-7; zmax = max(dip_zcoords)+100E-7
#         #     plt.quiver(dip_ycoords[:N], dip_zcoords[:N], p_perpart[:,0], p_perpart[:,1], pivot='mid', 
#         #         width=.5, #shaft width in arrow units 
#         #         scale=1., 
#         #         headlength=5,
#         #         headwidth=5.,
#         #         minshaft=4., 
#         #         minlength=.01
#         #         )
#         #     plt.xlim([ymin, ymax])
#         #     plt.ylim([zmin, zmax])
#             plt.axis('equal')
#             plt.yticks([]); plt.xticks([])
#         #     plt.scatter(dip_ycoords, dip_zcoords,c='blue',s=20)


# findmodes = NormalModes(
#         def_centers, # particle centers 
#         def_orient, # orientation of long axis prolate spheroid 
#         def_all_radii, # semi-major and semi-minor axis lengths of each spheroid
#         )
# findmodes.see_vectors()
# plt.show()


