from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

field_imag = loadmat('ez_imag')['ez_imag']
field_real = loadmat('ez_real')['ez_real']

fig = plt.figure(1, figsize=[7,3])
tot_phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
count=0
for i in tot_phases:
    count = count+1
    ax = fig.add_subplot(str(1)+str(len(tot_phases))+str(count), aspect='equal')

    phi=i
    Etot = field_real*np.cos(phi) + field_imag*np.sin(phi)
    plt.imshow(Etot,origin='lower',cmap='seismic')
    cbar_num_format = "%0.f"
    plt.clim([-4, 4])
    plt.title(str(np.int(phi*180/np.pi))+str('$^\circ$'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.1)
    cbar = plt.colorbar(format=cbar_num_format,cax=cax)
    ax.set_yticks([]); ax.set_xticks([])
fig.subplots_adjust(wspace=0.6)

plt.show()
