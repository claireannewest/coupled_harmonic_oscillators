clf;clc;
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' );
%  table of dielectric function
epstab = { epsconst( 1 ), epstable( 'drude.dat' ) };

%  diameter
diameter = 50;
%  nanosphere
p = comparticle( epstab, { trisphere( 256, diameter ) }, [ 2, 1 ], 1, op );

%  width of electron beam and electron velocity
[ width, vel ] = deal( 0.5, eelsbase.ene2vel( 200e3 ) );
%  impact parameter
imp = 20;
%  loss energies in eV
ene = linspace( 1., 3., 100 );

%  convert energies to nm
units;  enei = eV2nm ./ ene;

%  BEM solver
bem = bemsolver( p, op );
%  electron beam excitation
exc = electronbeam( p, [ diameter / 2 + imp, 0 ], width, vel, op );

%  surface loss
psurf = zeros( size( ene ) );

multiWaitbar( 'BEM solver', 0, 'Color', 'g', 'CanCancel', 'on' );
%  loop over energies
for ien = 1 : length( enei )
  %  surface charges
  sig = bem \ exc( enei( ien ) );
  %  EELS losses
  psurf( ien ) = exc.loss( sig );
  
  multiWaitbar( 'BEM solver', ien / numel( enei ) );
end
%  close waitbar
multiWaitbar( 'CloseAll' );

%  Mie solver
% mie = miesolver( epstab{ 2 }, epstab{ 1 }, diameter, op, 'lmax', 40 );

%  final plot
% plot( ene, psurf, 'o-', ene, mie.loss( imp, enei, vel ), '.-' );
plot( ene, psurf, 'o-');

% legend( 'BEM', 'Mie' );

xlabel( 'Loss energy (eV)' );
ylabel( 'Loss probability (eV^{-1})' );


write_it = [1240./enei; psurf];
fileID = fopen('Spectrum_eels_25nmsph','w');
fprintf(fileID,'%s %s \n', 'Energy [eV]', 'EELS');
fprintf(fileID,'%2.3f \t %2.5e \n',write_it);
fclose(fileID);
