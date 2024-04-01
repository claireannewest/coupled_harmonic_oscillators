clc;clear;
op = bemoptions( 'sim', 'ret', 'waitbar', 0 );
%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'drude.dat' ) };

%  light wavelength in vacuum
enei = linspace( 540, 2000, 200 );

short = 40; 
long = 100;

%  axes of ellipsoids
ax1 = [ 2*long, 2*short, 2*short];
ax2 = [ 2*long, 2*short, 2*short];
ax3 = [ 2*long, 2*short, 2*short];

%  nano ellipsoids
p1 = scale( trisphere( 144, 1 ), ax1 ); 
p2 = scale( trisphere( 144, 1 ), ax2 );
p3 = scale( trisphere( 144, 1 ), ax3 );

p1 = rot(p1, 0, [0, 0, 1]);
p2 = rot(p2, -60, [0, 0, 1]);
p3 = rot(p3, 60, [0, 0, 1]);


a = (2*long)*2.81; % length of equilateral triangle

p1 = shift(p1, [a/2, 0, 0] );
p2 = shift(p2, [3/4*a, sqrt(3)/4*a, 0] );
p3 = shift(p3, [1/4*a, sqrt(3)/4*a, 0] );

ptot = {p1, p2, p3};
p = comparticle( epstab, ptot, [ 2, 1; 2, 1; 2, 1], 1, 2, 3, op );

% k=x, E=y
%  set up BEM solver
bem = bemsolver( p, op );
%  plane wave excitation
plot(p)
exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0 ], op );
%%
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
%%
plot( 1240./enei, ext_mcsqrd, 'o-');  hold on;

xlabel( 'Energy (eV)' );
ylabel( 'Scattering cross section (nm^2)' );
%%
write_it = [1240./enei; ext_mcsqrd; abs_mcsqrd; sca_mcsqrd];
filename = strcat('Spectrum_trimer_40-100_a2.81_at144');
fileID = fopen(filename,'w');
fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it);
fclose(fileID);

beep on
beep
%%

enei(find(ext_mcsqrd == max(ext_mcsqrd)))
%% computation of electric field
bem = bemsolver( p, op );
exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0 ], op );

enei = enei(find(ext_mcsqrd == max(ext_mcsqrd)))
%  surface charge
sig = bem \ exc( p, enei );

%  mesh for calculation of electric field
[ x, y ] = meshgrid( linspace( -50, 600, 81 ) );
%  object for electric field
%    MINDIST controls the minimal distance of the field points to the
%    particle boundary
emesh = meshfield( p, x, y, 40, op, 'mindist', 0.9, 'nmax', 2000 );
%  induced and incoming electric field
e = emesh( sig ) + emesh( exc.field( emesh.pt, enei ) );
%  norm of electric field
beep on
beep

%%
ez_real = real(e(:,:,3));
ez_imag = imag(e(:,:,3));

max(ez_real, [], 'all')

imagesc( x( : ), y( : ), ez_real); 

colorbar;

colormap jet;

xlabel( 'x (nm)' );
ylabel( 'z (nm)' );

set( gca, 'YDir', 'norm' );
axis equal tight

%%
save('ez_real_a1.5.mat', 'ez_real')
save('ez_imag_a1.5.mat', 'ez_imag')

