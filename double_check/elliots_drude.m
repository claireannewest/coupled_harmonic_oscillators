clc; clear; clf;

w = linspace(0.5,4,101);
w(1) =[];

eps_inf = 9.695;
w_p = 8.959;
Gamma = 0.073;

Eps = eps_inf - w_p^2./(w.^2 + 1i*Gamma.*w);

n = sqrt((abs(Eps) + real(Eps))./2 );
k = sqrt((abs(Eps) - real(Eps))./2 );

Data = [w', n',k',real(Eps)',imag(Eps)'];
Data = sortrows(Data,1);

plot(1.24./Data(:,1),Data(:,2),...
     1.24./Data(:,1),Data(:,3));
%%

fid = fopen('drude.dat','wt');
fprintf(fid,' %s',' % drude model');
fprintf(fid,'\n');
fprintf(fid,' %s','% Energy eV  n  k');
fprintf(fid,'\n');

for j = 1:length(Data)
    fprintf(fid,' %g',Data(j,1));
    fprintf(fid,' %g',Data(j,2));
    fprintf(fid,' %g',Data(j,3));
    fprintf(fid,'\n');
end

fclose(fid);
