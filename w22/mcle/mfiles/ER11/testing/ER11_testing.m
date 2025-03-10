%ER11_testing.m

%%Dan Chavas, 2015-05-11

clear
close all
clc

cd '~/Dropbox/Research/MATLAB - PUBLIC/ER11E04_Chavas_PUBLIC_nondim/mfiles/ER11/testing'

addpath(genpath('../'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USER INPUT
Vmax = 50;  %[ms-1]
rmax_or_r0 = 'rmax';
r_in = 40*1000; %[m]
fcor = 5e-5;
CkCd = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Non-dimensional solution, no convergence to rmax,Vmax
tic
[rr,VV,~,~,~] = ER11_radprof_raw_nondim(Vmax,r_in,rmax_or_r0,fcor,CkCd);
toc

%% Non-dimensional solution, converged to rmax,Vmax
tic
rmax = r_in;
[rr_conv,VV_conv,rrfracrm_conv,MMfracMm_conv] = ER11_radprof_nondim(Vmax,rmax,fcor,CkCd);
toc

%% Non-dimensional solution, M/M0 vs. r/r0 only
tic
[rrfracrm,MMfracMm,rmax] = ER11_radprof_raw_nondim_MMm(Vmax,r_in,rmax_or_r0,fcor,CkCd);
toc

%% Dimensional solution
tic
[VV_compare,r0] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr);
rrfracrm_compare = rr/rmax;
Mm = .5*fcor*rmax.^2 + rmax.*Vmax;
MMfracMm_compare = (.5*fcor*rr.^2 + rr.*VV_compare)./Mm;
toc

%% PLOTTING
figure(1)
hold off
hpl(1) = plot(rrfracrm,MMfracMm,'b','LineWidth',2);
input_legend{1} = 'Nondim M soln';
hold on
plot(1,1,'r*','MarkerSize',14,'LineWidth',2)
hpl(2) = plot(rrfracrm_compare,MMfracMm_compare,'g--','LineWidth',2);
input_legend{2} = 'Dim V soln';
hpl(3) = plot(rrfracrm_conv,MMfracMm_conv,'m','LineWidth',1);
input_legend{3} = 'Nondim V soln converge (r_m,V_m)';
xlabel('r/r_m')
ylabel('M/M_m')
input_title = sprintf('r_m = %5.0f [km]',rmax/1000);
title(input_title)
legend(hpl,input_legend,'Location','NorthEast'); legend boxoff

%%Save plot
plot_filename = sprintf('ER11_testing_MMm.pdf');
saveas(gcf,plot_filename,'pdf')


figure(2)
hold off
hpl(1) = plot(rr/1000,VV,'b','LineWidth',2);
input_legend{1} = 'Nondim M soln';
hold on
plot(rmax/1000,Vmax,'r*','MarkerSize',14,'LineWidth',2)
hpl(2) = plot(rr/1000,VV_compare,'g--','LineWidth',2);
plot(r0/1000,0,'go','MarkerSize',14,'LineWidth',2)
input_legend{2} = 'Dim V soln';

hpl(3) = plot(rr_conv/1000,VV_conv,'m','LineWidth',1);
input_legend{3} = 'Nondim V soln converge (r_m,V_m)';

set(gca,'YLim',[0 Vmax])
xlabel('r [km]')
ylabel('V [m/s]')
input_title = sprintf('r_m = %5.0f [km]',rmax/1000);
title(input_title)
legend(hpl,input_legend,'Location','NorthEast'); legend boxoff

%%Save plot
plot_filename = sprintf('ER11_testing_V.pdf');
saveas(gcf,plot_filename,'pdf')
