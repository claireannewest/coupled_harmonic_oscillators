clc;%clf;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'drude.dat' ) };

%  diameter of sphere
diameter = 2*75;
%  initialize sphere
p = comparticle( epstab, { trisphere( 144, diameter ) }, [ 2, 1 ], 1, op );

%  set up BEM solver
bem = bemsolver( p, op );

%  plane wave excitation
exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0], op );
%  light wavelength in vacuum
enei = linspace( 450, 650, 200 );
%  allocate scattering and extinction cross sections
sca = zeros( length( enei ), 1 );
ext = zeros( length( enei ), 1 );

%  loop over wavelengths
for ien = 1 : length( enei ) 
  %  surface charge
  sig = bem \ exc( p, enei( ien ) );
  %  scattering and extinction cross sections
  sca( ien, : ) = exc.sca( sig );
  ext( ien, : ) = exc.ext( sig );
end
abs = ext - sca;
nmsqrd_to_micronsqrd = (10^(-6));
abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));
sca_mcsqrd = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));

plot( 1240./enei, abs_mcsqrd, 'o-'  );  hold on;

xlabel( 'Energy (eV)' );
ylabel( 'Absorption cross section (nm^2)' );

% write_it = [1240./enei; ext_mcsqrd; abs_mcsqrd; sca_mcsqrd];
% fileID = fopen('simulated_spectra/single_sphere/Spectrum_bemret_10nmsph144_drude_1.0','w');
% fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
% fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it);
% fclose(fileID);


%%  comparison with Mie theory
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
diameter = 2*(75);

mie = miesolver( epstable( 'drude.dat' ), epstab{ 1.0^2 }, diameter, op,'lmax',1);
abs = mie.ext( enei ) - mie.sca( enei );
plot( 1240./enei, abs*nmsqrd_to_micronsqrd);  hold on

% % legend( 'BEM : x-polarization', 'BEM : y-polarization', 'Mie theory' );
% abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
% write_it = [1240./enei; abs_mcsqrd];
% fileID = fopen('simulated_spectra/Spectrum_mie_50nmsph144_drude_1.0','w');
% fprintf(fileID,'%s %s \n', 'Energy [eV]', 'Abs Cross [um^2]');
% fprintf(fileID,'%2.3f \t %2.5e \n',write_it);
% fclose(fileID);
