%ER11E04_testing.m

%%Dan Chavas, 2015-05-11

clear
close all
clc

addpath(genpath('../../'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USER INPUT
Vmax = 50; 
r0 = 1000*1000;
fcor = 5e-5;
Cdvary = 1;
    C_d = 1.5e-3;
w_cool = 2e-3;
CkCdvary = 0;
    CkCd = 1;
eye_adj = 0;
    alpha_eye = .15;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Non-dimensional solution: r0 input
tic
[rr,VV,rmax,rmerge,Vmerge,rrfracr0,MMfracM0,rmaxr0,MmM0,rmerger0,MmergeM0] = ...
    ER11E04_nondim_r0input(Vmax,r0,fcor,Cdvary,C_d,w_cool,CkCdvary,CkCd,eye_adj,alpha_eye);
toc

%% Non-dimensional solution: rmax input
tic
[rr2,VV2,r02,rmerge2,Vmerge2,rrfracr02,MMfracM02,rmaxr02,MmM02,rmerger02,MmergeM02] = ...
    ER11E04_nondim_rmaxinput(Vmax,rmax,fcor,Cdvary,C_d,w_cool,CkCdvary,CkCd,eye_adj,alpha_eye);
toc

%% Dimensional solution
tic
rr_compare = (0:1:3000)*1000;   %[m]
ruser = r0;
Vuser = 0;
[VV_compare,rmax_compare,rmerge_compare,Vmerge_compare,~] = ER11E04_Chavas(Vmax,ruser,Vuser,fcor,Cdvary,C_d,w_cool,CkCd,rr_compare);
rrfracr0_compare = rr_compare/r0;
MMfracM0_compare = (.5*fcor*rr_compare.^2 + rr_compare.*VV_compare)./(.5*fcor*r0.^2);
rmerger0_compare = rmerge_compare/r0;
MmergeM0_compare = (rmerge_compare.*Vmerge_compare + .5*fcor*rmerge_compare.^2)/(.5*fcor*r0.^2);
toc
%}

%% PLOTTING
figure(1)
hold off
hpl(1) = plot(rrfracr0,MMfracM0,'b','LineWidth',2);
input_legend{1} = 'Nondim M soln: r_0 input';
hold on
plot(1,1,'b*','MarkerSize',14,'LineWidth',2)
plot(rmerger0,MmergeM0,'b*','MarkerSize',14,'LineWidth',2)
plot(rmaxr0,MmM0,'b*','MarkerSize',14,'LineWidth',2)

hpl(2) = plot(rrfracr02,MMfracM02,'g','LineWidth',1);
input_legend{2} = 'Nondim M soln: r_m input';
plot(1,1,'gx','MarkerSize',14,'LineWidth',1)
plot(rmerger0,MmergeM0,'gx','MarkerSize',14,'LineWidth',1)
plot(rmaxr0,MmM0,'gx','MarkerSize',14,'LineWidth',1)

hpl(3) = plot(rrfracr0_compare,MMfracM0_compare,'r--','LineWidth',1);
plot(rmerger0_compare,MmergeM0_compare,'rx','MarkerSize',14,'LineWidth',1)
input_legend{3} = 'Dim V soln';
xlabel('r/r_0')
ylabel('M/M_0')
input_title = sprintf('r_0 = %5.0f [km]',r0/1000);
title(input_title)
legend(hpl,input_legend,'Location','NorthWest'); legend boxoff

%%Save plot
plot_filename = sprintf('ER11E04_testing_MM0.pdf');
saveas(gcf,plot_filename,'pdf')


figure(2)
hold off
hpl(1) = plot(rr/1000,VV,'b','LineWidth',2);
input_legend{1} = 'Nondim M soln: r_0 input';
hold on
plot(r0/1000,0,'b*','MarkerSize',14,'LineWidth',2)
plot(rmerge/1000,Vmerge,'b*','MarkerSize',14,'LineWidth',2)
plot(rmax/1000,Vmax,'b*','MarkerSize',14,'LineWidth',2)

hpl(2) = plot(rr2/1000,VV2,'g','LineWidth',1);
input_legend{2} = 'Nondim M soln: r_m input';
plot(r02/1000,0,'gx','MarkerSize',14,'LineWidth',1)
plot(rmerge2/1000,Vmerge2,'gx','MarkerSize',14,'LineWidth',1)
plot(rmax/1000,Vmax,'gx','MarkerSize',14,'LineWidth',1)

hpl(3) = plot(rr_compare/1000,VV_compare,'r--','LineWidth',1);
plot(rmerge_compare/1000,Vmerge_compare,'gx','MarkerSize',14,'LineWidth',1)
input_legend{3} = 'Dim V soln';
set(gca,'YLim',[0 Vmax])
xlabel('r')
ylabel('V')
input_title = sprintf('r_0 = %5.0f [km]',r0/1000);
title(input_title)
legend(hpl,input_legend,'Location','NorthEast'); legend boxoff

%%Save plot
plot_filename = sprintf('ER11E04_testing_V.pdf');
saveas(gcf,plot_filename,'pdf')
