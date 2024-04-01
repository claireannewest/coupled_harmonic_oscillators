import numpy as np

class CoupledOscillators:
    def __init__(self, constants, num_part, num_dip, centers, orientations, radii, kind, **optional_semiaxis):
        """Defines the different system parameters.
        
        Keyword arguments:
        num_part -- number of particles 
        num_dip -- number of dipoles per particle (normally two)
        centers -- particle centers [cm]
        orientaitons -- unit vectors defining each dipole [unitless]
        radii -- radii of the particles [cm]
        kind -- which kind of dipole (0 = sphere, 1 = long axis prolate spheroid, 2 = short axis prolate spheroid)
        optional_semiaxis -- define the semi-axis of prolate sphereiod if it isn't defined in data file
        """
        self.c, self.hbar_eVs, self.e, self.wp, self.eps_inf, self.gamNR_qs, self.eps_b = constants
        self.num_part = num_part
        self.num_dip = num_dip 
        self.radii = radii
        self.centers = centers 
        self.unit_vecs = orientations
        self.kind = kind
        self.optional_semiaxis = optional_semiaxis

    def dipole_parameters(self):
        """Sets the physical dipole parameters. This is assuming the dipoles represent spheres or prolate spheroids.
        """
        mat_size = int(self.num_dip*self.num_part)
        w0 = np.zeros(mat_size) # initiaize array for resonance frequency for each dipole
        m = np.zeros(mat_size) # initiaize array for effective for each dipole
        gamNR = np.zeros(mat_size) # initiaize array for nonradiative damping for each dipole
        n = np.sqrt(self.eps_b)
        for row in range(0, mat_size):
            if self.kind[row] == 0: # sphere
                D = 1.
                li = self.radii[row]
                v = 4./3*np.pi*(self.radii[row])**3 # [cm^3], volume of sphere
                Li = 1./3
                w0_qs = np.sqrt(self.wp**2/(self.eps_inf+2*self.eps_b)) # [eV], quasi-static plasmon resonance frequency
                m_qs = 4.*np.pi*self.e**2/(3*v*(w0_qs/self.hbar_eVs)**2)*(self.eps_inf + 2*self.eps_b)/(3*self.eps_b)
                # Long wavelength approximation (https://www.osapublishing.org/josab/viewmedia.cfm?uri=josab-26-3-517&seq=0)
                m[row] =  m_qs + D*self.e**2/(li*self.c**2) # [g], mass with radiation damping
                w0[row] = w0_qs*np.sqrt(m_qs/m[row]) # [eV], plasmon resonance frequency with radiation damping
                gamNR[row] = self.gamNR_qs * m_qs/m[row] # [eV], adjusted nonradiative damping

            if self.kind[row] == 1: # prolate spheroid
                cs = self.radii[row] # [cm], semi-major axis of prolate sphereiod 
                # find which row corresponds to the semi-minor axis of prolate spheroid (this is required to calculate dipole parameters)
                idx = np.where( (self.centers[:,0] == self.centers[row,0]) &  # the semi-minor axis will have the same origin
                        (self.centers[:,1] == self.centers[row,1]) & # the semi-minor axis will have the same origin
                        (self.kind == 2) # the semi-minor axis will have kind = 2
                        )
                if self.optional_semiaxis == '': 
                    a = self.radii[idx] # [cm], semi-minor axis of prolate spheriod 
                if self.optional_semiaxis != '': 
                    a = self.optional_semiaxis # [cm], semi-minor axis of prolate spheriod 
                es = np.sqrt((cs**2 - a**2)/cs**2) # [unitless]
                V = 4./3*np.pi*a**2*cs # [cm^3]
                Lz = (1-es**2)/es**2*(-1+1/(2*es)*np.log((1+es)/(1-es))) # [unitless]
                Dz = 3./4*((1+es**2)/(1-es**2)*Lz + 1) # [unitless]
                Ly = (1-Lz)/2 # [unitless]  
                Dy = a/(2*cs)*( 3/es*np.arctanh(es) - Dz) 
                # for semi-major axis
                lE = cs; Li = Lz; D = Dz
                w0_qs = np.sqrt((wp**2/eps_b)/((eps_inf/eps_b-1)+1/Li)) # [eV], quasi-static plasmon resonance frequency 
                m_qs= 4*np.pi*e**2*((eps_inf/eps_b-1)+1/Li)/((w0_qs/hbar_eVs)**2*V/Li**2) # [g] 
                m[row] = m_qs + n**2*Dz*e**2/(lE*c**2) # [g]
                w0[row] = (w0_qs)*np.sqrt(m_qs/m[row]) # [eV]
                gamNR[row] = 0.07*(m_qs/m[row]) # [eV]
                # for semi-minor axis 
                lE = a; Li = Ly; D = Dy
                w0_qs = np.sqrt((wp**2/eps_b)/((eps_inf/eps_b-1)+1/Li)) # [eV], quasi-static plasmon resonance frequency 
                m_qs= 4*np.pi*e**2*((eps_inf/eps_b-1)+1/Li)/((w0_qs/hbar_eVs)**2*V/Li**2) # g 
                m[idx] = m_qs + n**2*Dy*e**2/(lE*c**2) # g (charge and speed of light)
                w0[idx] = (w0_qs)*np.sqrt(m_qs/m[idx]) # 1/s
                gamNR[idx] = 0.07*(m_qs/m[idx]) 
        return w0, m, gamNR # [eV], [g], [eV]
    
    def gamma(self, w):
        w0, m, gamNR = self.dipole_parameters()
        n = np.sqrt(self.eps_b)
        gam_rad = n**3 * w**2 * (2.0*self.e**2) / (3.0*m*self.c**3)/self.hbar_eVs
        return gamNR, gam_rad

    def alpha(self, w):
        w0, m, gamNR = self.dipole_parameters()
        gamTot = self.gamma(w)[0] + self.gamma(w)[1]
        return self.e**2 / m * self.hbar_eVs**2 / (-w**2 - 1j * gamTot * w + w0**2)

    def cross_sections(self, w):
        w0, m, gamNR = self.dipole_parameters()
        gamNR, gamR = self.gamma(w)
        gamTot = gamNR + gamR
        abs_cross = 4.*np.pi*w/self.c*(gamNR/gamTot)*np.imag(self.alpha(w))
        sca_cross = 4.*np.pi*w/self.c*(gamR/gamTot)*np.imag(self.alpha(w))
        ext_cross = 4.*np.pi*w/self.c*np.imag(self.alpha(w))
        return ext_cross, abs_cross, sca_cross
























    def coupling(self, dip_i, dip_j, k): 
        """Calculates the off diagonal matrix elements, which is the 
        coupling between the ith and jth dipole divided by the effective 
        mass of the ith dipole.
        
        Keyword arguments:
        dip_i -- ith dipole
        dip_j -- jth dipole
        k -- wave vector [1/cm]
        """
        c, hbar_eVs, e, wp, eps_inf, gamNR_qs, eps_b = self.constants
        # k = np.real(k) # [1/cm]
        r_ij = self.centers[dip_i,:] - self.centers[dip_j,:] # [cm], distance between ith and jth dipole 
        mag_rij = np.linalg.norm(r_ij) 
        if mag_rij == 0: 
        # If i and j are at the same location, the coupling (g) is zero (prevents a divide by zero error.)
            g=0
        else:
            nhat_ij = r_ij / mag_rij # [unitless], unit vector in r_ij direction.
            xi_hat = self.unit_vecs[dip_i] # [unitless], unit vector of the ith dipole
            xj_hat = self.unit_vecs[dip_j] # [unitless], unit vector of the jth dipole
            xi_dot_nn_dot_xj = np.dot(xi_hat, nhat_ij)*np.dot(nhat_ij, xj_hat) # [unitless]
            nearField = ( 3.*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat) ) / mag_rij**3 # [1/cm^3], near field coupling.
            intermedField = -1j*k*(3*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij**2 # [1/cm^3], intermediate field coupling
            farField = k**2*(xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij # [1/cm^3], far field coupling
            g = e**2 * hbar_eVs**2 * ( nearField  - intermedField - farField ) * np.exp(1j*k*mag_rij) # [g eV^2], total radiative coupling
        return g # [g eV^2]
    
    def make_matrix(self, k):
        """Forms the matrix A for A(w)*x = w^2*x. 
        
        Keywords: 
        k -- wave vector [1/cm]
        """
        c, hbar_eVs, e, wp, eps_inf, gamNR_qs, eps_b = self.constants
        n = np.sqrt(eps_b)
        mat_size = int(self.num_dip*self.num_part)
        matrix = np.zeros( (mat_size, mat_size) ,dtype=complex) 
        w_guess = k*c/np.sqrt(eps_b)*hbar_eVs # [eV], left-hand size w 
        gam = self.gamNR + n**3*w_guess**2*(2.0*e**2)/(3.0*self.m*c**3)/hbar_eVs # [eV], radiative damping for dipole i
        matrix[( np.arange(mat_size), np.arange(mat_size) )] = self.w0**2 - 1j*gam*w_guess # [eV^2], on-diagonal matrix elements
        for dip_i in range(0 , mat_size): 
            for dip_j in range(0, mat_size): 
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
        c, hbar_eVs, e, wp, eps_inf, gamNR_qs, eps_b = self.constants
        mat_size = int(self.num_dip*self.num_part)
        prec = 10 # convergence condition for iteration
        final_eigvals = np.zeros(mat_size,dtype=complex) 
        final_eigvecs = np.zeros( (mat_size, mat_size), dtype=complex) 
        w_init = -1j*self.gamNR/2. + np.sqrt(-self.gamNR**2/4.+self.w0**2) # [eV], Initial guess of left-hand size w. This is the result for an isolated nonradiative dipole.
        for mode in range(0,mat_size): # Converge each mode individually.
            eigval_hist = np.array([w_init[mode], w_init[mode]*1.1],dtype=complex) 
            eigvec_hist = np.zeros((mat_size, 2))
            count = 0
            blow_up_count = 0
            while (np.abs((np.real(eigval_hist[0]) - np.real(eigval_hist[1])))  > 10**(-prec)) or \
                  (np.abs((np.imag(eigval_hist[0]) - np.imag(eigval_hist[1]))) > 10**(-prec)) or \
                  np.all((np.abs((np.real(eigvec_hist[:,0]) - np.real(eigvec_hist[:,1]))) > 10**(-prec))) or \
                  np.all((np.abs((np.imag(eigvec_hist[:,0]) - np.imag(eigvec_hist[:,1]))) > 10**(-prec))):
                w_guess = eigval_hist[0]
                if count > 100: 
                    denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
                    w_guess = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom 
                k = w_guess/hbar_eVs*np.sqrt(eps_b)/c # [1/cm]

                if w_guess > 10: 
                    # if you're starting to blow up (i.e. w > 10 eVs), then give up and start over
                    # if you've entereted this loop more than once, you'll need new guesses
                    w0, m, gamNR = self.dipole_parameters()
                    ## pick a_rand, b_rand, c_rand to be a random number between 0.8 - 1.2
                    import random
                    a_rand = random.uniform(0.8, 1.2)
                    b_rand = random.uniform(0.8, 1.2)
                    c_rand = random.uniform(0.8, 1.2)

                    eigval_hist = np.array([w0[mode]*a_rand, w0[mode]*b_rand, w0[mode]*c_rand],dtype=complex) 
                    eigvec_hist = np.zeros((mat_size, 3))
                    continue


                val, vec, H = self.make_matrix(k=k)
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
            final_eigvals[mode] = eigval_hist[0]
            final_eigvecs[:,mode] = eigvec_hist[:,0]
        return final_eigvals, final_eigvecs  
    
    def see_vectors(self, final_eigvals, final_eigvecs):
        """Plot the convereged eigenvectors."""
        dip_ycoords = self.centers[:,0]
        dip_zcoords = self.centers[:,1] 
        mat_size = int(self.num_dip*self.num_part) 
        plt.figure(1, figsize=[6,1.5])
        for mode in range(0, mat_size):
            w = final_eigvals[mode] # [eV]
            energy = np.real(w)
            # wamp = np.abs(w)
            # phi = np.arctan2(np.imag(w), np.real(w))
            # energy = wamp#*(np.cos(phi)+1j*np.sin(phi))

            # #####
            # v = final_eigvecs[:,mode] # [unitless]
            # phi = np.arctan2(np.imag(v), np.real(v))
            # # print('phi = ', phi)

            # vamp = np.abs(v)
            # vector_mag = np.real(vamp*(np.cos(phi)+1j*np.sin(phi)))
            v = np.real(final_eigvecs[:,mode]) # [unitless]
            vector_mag = v
            plt.subplot(1,mat_size,mode+1)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.title('%.2f' % energy + ' eV', fontsize=10)
            p = vector_mag[...,np.newaxis]*self.unit_vecs
            if self.num_dip == 1: p_perpart = p
            else: p_perpart = p[:int(mat_size/self.num_dip),:] + p[int(mat_size/self.num_dip):,:] 
            ymin = min(dip_ycoords)-100E-7; ymax = max(dip_ycoords)+100E-7
            zmin = min(dip_zcoords)-100E-7; zmax = max(dip_zcoords)+100E-7
            plt.quiver(dip_ycoords[:int(mat_size/self.num_dip)], dip_zcoords[:int(mat_size/self.num_dip)], p_perpart[:,0], p_perpart[:,1], pivot='mid', 
                width=.5, #shaft width in arrow units 
                scale=1., 
                headlength=5,
                headwidth=5.,
                minshaft=4., 
                minlength=.01
                )
            plt.yticks([]); plt.xticks([])
            plt.scatter(dip_ycoords, dip_zcoords,c='blue',s=20)
            plt.xlim([ymin, ymax])
            plt.ylim([zmin, zmax])
            plt.subplots_adjust(top=.83)

