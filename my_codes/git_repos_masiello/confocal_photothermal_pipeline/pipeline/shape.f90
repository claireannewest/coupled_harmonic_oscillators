program shapemaker

    ! The following sample code creates two rods.
    real :: r, t, A, C, B, DS
    integer :: x,y,z,idx, x_thick, y_length, z_width, y_offset, z_offset, rastery, rasterz
    character(len=200)::row

    open(12, file='temp3',status='replace')

    !!~~~~~ Parameters to edit ~~~~~!! 
    DS = 2 ! Sets units of the lattice (DS = 2 corresponds to a lattice spacing of 2 nm)
    r = 4 ! Sets curvature in x, y direction. Larger number = less curved. Must be > 1
    t = 4 ! Sets curvature in y, z direction Larger number = less curved. Must be > 1   
    gap = 20/DS ! Edge-to-edge distance separating the two rods
    x_thick = 40 ! Thickness along optical axis of both rods, units in nm
    y_length  = 80 ! Length in y direction of both rods, units in nm
    z_width = 40 ! Length in z direction of both rods, units in nm

    !!~~~~~ Defining parameters (do not edit) ~~~~~!!                                                                                 
    rastery = 0 ! y offset that will be looped in order to raster 
    rasterz = 0 ! z offset that will be looped in order to raster
    idx = 0 ! index for counting number of dipoles
    A = x_thick/(2*DS)+1 ! semi-diameter in x direction
    B = y_length/(2*DS)+1 ! semi-diameter in y direction 
    C = z_width/(2*DS)+1 ! semi-diameter in z direction 

    !!~~~~~ Defining rod one ~~~~~!!
    y_offset = -y_length/2/DS - gap/2 + rastery ! Y value of center of rod
    z_offset = rasterz ! Z value of center of rod   

    do x = -x_thick/2, x_thick/2
       do y = -y_length/2,y_length/2
          do z = -z_width/2, z_width/2
             ! Equation of a superellipsoid generates a rectangle with rounded corners
             if ( (ABS((x/A))**r + ABS((y/B))**r)**(t/r) + ABS((z/C))**t  < 1 ) then
                idx = idx+1
                write(12,*) idx , INT(x-A) , INT(y+y_offset) , INT(z+z_offset) ,1,1,1
             end if
          end do
       end do
    end do

   !!~~~~~ Defining rod two ~~~~~!!
    y_offset = y_length/2/DS + gap/2 + rastery	! Y value of center of rod
    z_offset = rasterz ! Z value of center of rod
    do x = -x_thick/2, x_thick/2
       do y = -y_length/2,y_length/2
          do z = -z_width/2, z_width/2
             ! Equation of a superellipsoid generates a rectangle with rounded corners
             if ( (ABS((x/A))**r + ABS((y/B))**r)**(t/r) + ABS((z/C))**t  < 1 ) then
                idx = idx+1
                write(12,*) idx , INT(x-A) , INT(y+y_offset) , INT(z+z_offset) ,1,1,1
             end if
          end do
       end do
    end do

    close(12)
    open(12, file='temp3',status='old')
    open(13, file='shape.dat',status='replace')
    write(13,*) 'Sphere shape'
    write(13,*) idx, '= number of dipoles in target'
    write(13,*) '1.000000 0.000000 0.000000 = A_1 vector'
    write(13,*) '0.000000 1.000000 0.000000 = A_2 vector'
    write(13,*) '1.000000 1.000000 1.000000 = (d_x,d_y,d_z)/d'
    write(13,*) '0.000000 0.000000 0.000000 = (x,y,z)/d'
    write(13,*) 'JA  IX  IY  IZ ICOMP(x,y,z)'

do x = 1,idx
   read(12,'(a)') row 
   write(13,'(a)') trim(row)
end do
close(12,status='delete')
close(13)
end program shapemaker
