%E04_outerwind_r0input_testing.m

%%Dan Chavas, 2015-05-11

clear
close all
clc

addpath(genpath('../'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USER INPUT
r0 = 2133*1000;  %[m]
fcor = 5e-5;    %[s-1]
Cdvary = 0;
    C_d = 1.5e-3;   %[-]
w_cool = 2e-3;  %[m/s]
Nr = 10000;    %[-]; set to large number to integrate all the way to near 0  
V_max = 100;    %[ms-1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gam = C_d*fcor*r0/w_cool

%% Non-dimensional solution
tic
[rr,VV,rrfracr0,MMfracM0] = E04_outerwind_r0input_nondim(r0,fcor,Cdvary,C_d,w_cool,Nr);
toc

%% Dimensional solution
% tic
% [rr_compare,VV_compare] = E04_outerwind_r0input(r0,fcor,Cdvary,C_d,w_cool,V_max);
% rrfracr0_compare = rr_compare/r0;
% MMfracM0_compare = (.5*fcor*rr_compare.^2 + rr_compare.*VV_compare)./(.5*fcor*r0.^2);
% toc

%% PLOTTING
figure(1)
hold off
hpl(1) = plot(rrfracr0,MMfracM0,'b','LineWidth',2);
input_legend{1} = 'Nondim M soln';
hold on
plot(1,1,'r*','MarkerSize',14,'LineWidth',2)
% hpl(2) = plot(rrfracr0_compare,MMfracM0_compare,'g--','LineWidth',2);
% input_legend{2} = 'Dim V soln';
xlabel('r/r_0')
ylabel('M/M_0')
input_title = sprintf('gam = %3.1f',gam);
title(input_title)
legend(hpl,input_legend,'Location','NorthWest'); legend boxoff

%%Save plot
plot_filename = sprintf('E04_outerwind_r0input_testing_MM0.pdf');
saveas(gcf,plot_filename,'pdf')


figure(2)
hold off
hpl(1) = plot(rr/1000,VV,'b','LineWidth',2);
input_legend{1} = 'Nondim M soln';
hold on
plot(r0/1000,0,'r*','MarkerSize',14,'LineWidth',2)
% hpl(2) = plot(rr_compare/1000,VV_compare,'g--','LineWidth',2);
% input_legend{2} = 'Dim V soln';
set(gca,'YLim',[0 V_max])
xlabel('r')
ylabel('V')
input_title = sprintf('gam = %3.1f; r_0 = %5.0f [km]',gam,r0/1000);
title(input_title)
legend(hpl,input_legend,'Location','NorthWest'); legend boxoff

%%Save plot
plot_filename = sprintf('E04_outerwind_r0input_testing_V.pdf');
saveas(gcf,plot_filename,'pdf')
