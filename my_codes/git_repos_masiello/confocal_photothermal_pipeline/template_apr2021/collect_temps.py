import numpy as np
import os.path

ybegin=-200
yend=201
stepsize=10
x_plane = -11

zbegin=-200
zend=201
stepsize=10

file = open(str('scatter_hot'),'w')
file.write(str('Wave') + '\t' + str('x') + '\t' + str('y') + '\t' + str('z') + '\t' + str('C_sca [um^2]') + '\n')
for valy in range(ybegin,yend,stepsize):
    for valz in range(zbegin,zend,stepsize): 
        filenameH = str('x00_y') + str(valy) + str('_z') + str(valz) + str('/Integration_f11f11_hot')
        if os.path.isfile(filenameH) == True and os.stat(filenameH).st_size != 81:
            with open(filenameH) as fileH:
                for lineH in fileH:
                    pass
                last_lineH = lineH
                wave_um = last_lineH.split()[0]
                c_sca = last_lineH.split()[5]
                shape_offset_x = 0
                shape_offset_y = valy
                shape_offset_z = valz
                file.write(str(wave_um) + '\t' + str(shape_offset_x) + '\t' + str(shape_offset_y) + '\t' + str(shape_offset_z) + '\t' + str(c_sca) + '\n')
        else:
            print(str(filenameH), (' did not converge'))
file.close()

file = open(str('scatter_room'),'w')
file.write(str('Wave') + '\t' + str('x') + '\t' + str('y') + '\t' + str('z') + '\t' + str('C_sca [um^2]') + '\n')
for valy in range(ybegin,yend,stepsize): 
    for valz in range(zbegin,zend,stepsize):
        filenameH = str('x00_y') + str(valy) + str('_z') + str(valz) + str('/Integration_f11f11_hot')
        filenameR = str('x00_y') + str(valy) + str('_z') + str(valz) + str('/Integration_f11f11_room')
        if os.path.isfile(filenameR) == True and os.stat(filenameH).st_size != 81:
            with open(filenameR) as fileR:
                for lineR in fileR:
                    pass
                last_lineR = lineR
                wave_um = last_lineR.split()[0]
                c_sca = last_lineR.split()[5]
                shape_offset_x = 0
                shape_offset_y = valy
                shape_offset_z = valz
            file.write(str(wave_um) + '\t' + str(shape_offset_x) + '\t' + str(shape_offset_y) + '\t' + str(shape_offset_z) + '\t' + str(c_sca) + '\n')
        else:
            print(str(filename), (' did not converge'))
file.close()

# file = open(str('particle_temps.txt'),'w')                                                       
# file.write(str('Y center') + '\t' + str('Z center') + '\t'+ str('Part 1') + '\n')
# for valy in range(ybegin,yend,stepsize):
#     for valz in range(zbegin,zend,stepsize):
#         filename = str('x00_y') + str(valy) + str('_z') + str(valz) + str('/temp.out')
#         data = np.loadtxt(filename)
#         part1_idx = data[np.where( (data[:,0] == x_plane) & (data[:,1] == -25+valy) & ( data[:,2] == -10+valz ))]
#         part2_idx = data[np.where( (data[:,0] == x_plane) & (data[:,1] == 25+valy) & ( data[:,2] == -10+valz ))]
#         part3_idx = data[np.where( (data[:,0] == x_plane) & (data[:,1] == -6+valy) & ( data[:,2] == 10+valz ))]
#         file.write(str(valy) + '\t' + str(valz) + '\t' + str(part1_idx[:,3][0]) + '\t' + str(part2_idx[:,3][0]) + '\t' + str(part3_idx[:,3][0]) +'\n')
# file.close()
