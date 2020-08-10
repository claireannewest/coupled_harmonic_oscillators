clf;clc;
op = bemoptions( 'sim', 'ret', 'waitbar', 0 );
%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'drude.dat' ) };

%  light wavelength in vacuum
enei = linspace( 1240/3.0, 1240/1.5, 300 );

radx1 = 20;
rady1 = 20; 
radz1 = 70;

radx2 = 20; 
rady2 = 20;
radz2 = 50;

%  axes of ellipsoids
ax1 = [ 2*radx1, 2*rady1, 2*radz1];
ax2 = [ 2*radx2, 2*rady2, 2*radz2];

% gapx = 0;
% gapy = 0;
gapz = 10;

%  nano ellipsoids
p1 = scale( trisphere( 144, 1 ), ax1 );
p2 = scale( trisphere( 144, 1 ), ax2 );

% shift ellipsoids 
p1 = shift(p1, [0, 0, -radz1-gapz/2] );
p2 = shift(p2, [0, 0,  radz2+gapz/2] );


p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1], 1, 2, op );

%  set up BEM solver
bem = bemsolver( p, op );
%  plane wave excitation
exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0 ], op );

%  loop over wavelengths
for ien = 1 : length( enei )
%  surface charge
sig = bem \ exc( p, enei( ien ) );
%  scattering and extinction cross sections
sca( ien, 1 ) = exc.sca( sig );
ext( ien, 1 ) = exc.ext( sig );
end

abs = ext - sca;
nmsqrd_to_micronsqrd = (10^(-6));
abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));
sca_mcsqrd = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));

plot( 1240./enei, abs_mcsqrd, 'o-');  hold on;

xlabel( 'Wavelength (nm)' );
ylabel( 'Scattering cross section (nm^2)' );
%%
write_it = [1240./enei; ext_mcsqrd; abs_mcsqrd; sca_mcsqrd];
fileID = fopen('simulated_spectra/two_ellipsoids/Spectrum_bemret_202070_202050_10_drude_1.0_y','w');
fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it);
fclose(fileID);
