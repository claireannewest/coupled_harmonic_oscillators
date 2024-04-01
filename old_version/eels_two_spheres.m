clf;clc;
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' );
%  table of dielectric function
epstab = { epsconst( 1 ), epstable( 'drude.dat' ) };

%  size of spheres
radii1 = 25;
radii2 = 25;

start = 1000;
endit = 1200;
nums = (endit-start)/10+1;

gaprange = linspace( start, endit, nums );

for gap = 1 : length( gaprange )
    gap = gaprange(gap)

    % initialize spheres
    diameter1 = 2*radii1;
    diameter2 = 2*radii2;
    p1 = trisphere( 144, diameter1 );
    p2 = trisphere( 144, diameter2 );

    % shift spheres
    p1 = shift(p1, [0, -gap/2, 0] );
    p2 = shift(p2, [0, gap/2, 0] );

    %  initialize sphere
    p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1 ], 1, 2, op );

    %  width of electron beam and electron velocity
   [ width, vel ] = deal( 0.5, eelsbase.ene2vel( 200e3 ) );
    %  impact parameter
    imp = 20;
    %  loss energies in eV
    ene = linspace( 2.5, 2.7, 100 );

    %  convert energies to nm
    units;  enei = eV2nm ./ ene;


    %%%%  BEM solver bonding %%%%
    bem = bemsolver( p, op );
    %  electron beam excitation
    exc_bond = electronbeam( p, [ gap/2+radii1+10, 0 ], width, vel, op );
    %  surface loss
    psurf_bond = zeros( size( ene ) );
    multiWaitbar( 'BEM solver', 0, 'Color', 'g', 'CanCancel', 'on' );
    %  loop over energies
    for ien = 1 : length( enei )
      %  surface charges
      sig = bem \ exc_bond( enei( ien ) );
      %  EELS losses
      psurf_bond( ien ) = exc_bond.loss( sig );

      multiWaitbar( 'BEM solver', ien / numel( enei ) );
    end

    %%%%  BEM solver antibonding %%%%
    bem = bemsolver( p, op );
    %  electron beam excitation
    exc_anti = electronbeam( p, [ 0, 0 ], width, vel, op );

%     exc_anti = electronbeam( p, [ gap/2-radii1-10, 0 ], width, vel, op );
    %  surface loss
    psurf_anti = zeros( size( ene ) );
    multiWaitbar( 'BEM solver', 0, 'Color', 'g', 'CanCancel', 'on' );
    %  loop over energies
    for ien = 1 : length( enei )
      %  surface charges
      sig = bem \ exc_anti( enei( ien ) );
      %  EELS losses
      psurf_anti( ien ) = exc_anti.loss( sig );

      multiWaitbar( 'BEM solver', ien / numel( enei ) );
    end
    multiWaitbar( 'CloseAll' );


    plot( ene, psurf_bond );
    plot( ene, psurf_anti, 'go-');

    % legend( 'bond', 'anti' );
    xlabel( 'Loss energy (eV)' );
    ylabel( 'Loss probability (eV^{-1})' );

    filename = strcat('Spectrum_eels_25nmsph_gap', string(gap));

    write_it = [1240./enei; psurf_bond; psurf_anti];
    fileID = fopen(filename,'w');
    fprintf(fileID,'%s %s %s \n', 'Energy [eV]', 'EELS_b', 'EELS_a');
    fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \n',write_it);
    fclose(fileID);

    % plot( ene, psurf_bond/max(psurf_bond), 'bo-' ); hold on;
    % plot( ene, psurf_anti/max(psurf_anti), 'go-');
end
