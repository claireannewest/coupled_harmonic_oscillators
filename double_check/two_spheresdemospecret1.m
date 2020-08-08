%  DEMOSPECRET1 - Light scattering of metallic nanosphere.
%    For a metallic nanosphere and an incoming plane wave, this program
%    computes the scattering cross section for different light wavelengths
%    using the full Maxwell equations, and compares the results with Mie
%    theory.
%
%  Runtime on my computer:  7.4 sec.

%%  initialization
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' );

%  table of dielectric functions
% epstab = { epsconst( 1 ), epstable( 'gold.dat' ) };
epstab = { epsconst( 1.473^2 ), epsdrude( 'Au' ) };
%  diameter of sphere
a1 = 10;
a2 = 15;

diameter1 = a1*2;
diameter2 = a2*2;
ccsep = 27;
%  initialize sphere
p1 = trisphere( 300, diameter1 );
p2 = shift( trisphere( 300, diameter2 ), [ ccsep, 0, 0 ] );

p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1 ], 1, 2, op );
figure()
% 
plot(p, 'EdgeColor', 'b')
%%  BEM simulation
%  set up BEM solver
bem = bemsolver( p, op );

%  plane wave excitation
exc = planewave( [ 1, 0, 0; 0, 1, 0 ], [ 0, 0, 1; 0, 0, 1 ], op );
%  light wavelength in vacuum
enei = eV2nm ./ linspace( 1.8, 2.6, 100 );
%  allocate scattering and extinction cross sections
sca = zeros( length( enei ), 2 );
ext = zeros( length( enei ), 2 );

multiWaitbar( 'BEM solver', 0, 'Color', 'g', 'CanCancel', 'on' );
%  loop over wavelengths
for ien = 1 : length( enei )
  %  surface charge
  sig = bem \ exc( p, enei( ien ) );
  %  scattering and extinction cross sections
  sca( ien, : ) = exc.sca( sig );
  ext( ien, : ) = exc.ext( sig );
  
  multiWaitbar( 'BEM solver', ien / numel( enei ) );
end
%  close waitbar
multiWaitbar( 'CloseAll' );

%%  final plot
figure()

plot( 1240./enei, ext-sca, 'o-'  );  hold on;

xlabel( 'Wavelength (nm)' );
ylabel( 'Absorption cross section (nm^2)' );

% %%  comparison with Mie theory
% mie = miesolver( epstab{ 2 }, epstab{ 1 }, diameter, op );
% 
% plot( 1240./enei, mie.abs( enei ), '--' );  hold on
% 
legend( 'BEM : x-polarization', 'BEM : y-polarization' );
