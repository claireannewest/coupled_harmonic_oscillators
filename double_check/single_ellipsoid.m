% clf;clc;
op = bemoptions( 'sim', 'stat', 'waitbar', 0 );
%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'drude.dat' ) };

%  light wavelength in vacuum
enei = linspace( 450, 600, 200 );

%  axes of ellipsoids
ax = [ 2*5, 2*5, 2*10];

%  loop over different ellipsoids
for i = 1 : size( ax, 1 )
  
  %  nano ellipsoid
  p = scale( trisphere( 144, 1 ), ax( i, : ) );
  %  set up COMPARTICLE object
  p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

  %  set up BEM solver
  bem = bemsolver( p, op );
  %  plane wave excitation
  exc = planewave( [ 0, 0, 1 ], [ 1, 0, 0 ], op );

  %  loop over wavelengths
  for ien = 1 : length( enei )
    %  surface charge
    sig = bem \ exc( p, enei( ien ) );
    %  scattering and extinction cross sections
    sca( ien, i ) = exc.sca( sig );
    ext( ien, i ) = exc.ext( sig );
  end

end
abs = ext - sca;
nmsqrd_to_micronsqrd = (10^(-6));
abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));
sca_mcsqrd = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));

plot( 1240./enei, abs_mcsqrd, 'o-');  hold on;

xlabel( 'Wavelength (nm)' );
ylabel( 'Scattering cross section (nm^2)' );

write_it = [1240./enei; ext_mcsqrd; abs_mcsqrd; sca_mcsqrd];
fileID = fopen('simulated_spectra/single_ellipsoid/Spectrum_bemstat_5_5_10_drude_1.0','w');
fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it);
fclose(fileID);

