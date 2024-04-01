clf;clc;
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' );
%  table of dielectric function
epstab = { epsconst( 1 ), epstable( 'drude.dat' ) };


start = 50;
endit = 55;
nums = (endit-start)/10+1;
gaprange = linspace( start, endit, nums );
gaprange=50

for gap = 1 : length( gaprange )
    gap = gaprange(gap)
    
    radx1 = 5;
    rady1 = 20; 
    radz1 = 5;

    radx2 = 5; 
    rady2 = 20;
    radz2 = 5;

    %  axes of ellipsoids
    ax1 = [ 2*radx1, 2*rady1, 2*radz1];
    ax2 = [ 2*radx2, 2*rady2, 2*radz2];

    %  nano ellipsoids
    p1 = scale( trisphere( 144, 1 ), ax1 );
    p2 = scale( trisphere( 144, 1 ), ax2 );
    
    % shift ellipsoids
    p1 = shift(p1, [0, -gap/2, 0] );
    p2 = shift(p2, [0, gap/2, 0] );

    %  initialize sphere
    p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1 ], 1, 2, op );
    plot(p)
    %  width of electron beam and electron velocity
   [ width, vel ] = deal( 0.5, eelsbase.ene2vel( 200e3 ) );
    %  impact parameter
    imp = 10;
    %  loss energies in eV
    ene = linspace( 1.5, 2.5, 100 );

    %  convert energies to nm
    units;  enei = eV2nm ./ ene;


    %%%%  BEM solver bonding %%%%
    bem = bemsolver( p, op );
    %  electron beam excitation
    exc_bond = electronbeam( p, [0, gap/2+radii1+10], width, vel, op );
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


    plot( ene, psurf_bond ); hold on;
    plot( ene, psurf_anti, 'go-');

    % legend( 'bond', 'anti' );
    xlabel( 'Loss energy (eV)' );
    ylabel( 'Loss probability (eV^{-1})' );

    filename = strcat('Spectrum_eels_5-20ellipsoid_gap', string(gap));

    write_it = [1240./enei; psurf_bond; psurf_anti];
    fileID = fopen(filename,'w');
    fprintf(fileID,'%s %s %s \n', 'Energy [eV]', 'EELS_b', 'EELS_a');
    fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \n',write_it);
    fclose(fileID);

    % plot( ene, psurf_bond/max(psurf_bond), 'bo-' ); hold on;
    % plot( ene, psurf_anti/max(psurf_anti), 'go-');
end
