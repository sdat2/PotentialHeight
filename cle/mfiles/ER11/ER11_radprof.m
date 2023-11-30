%ER11_radprof.m -- ER11_radprof converged on correct (Vmax,r_in)
%
% Syntax:  [V_ER11,r_out] = ER11_radprof(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11)
%
% Inputs:
%   Vmax [ms-1] - maximum wind speed
%   r_in [m] - either radius of maximum wind (rmax) or outer radius of
%      vanishing wind (r0)
%   rmax_or_r0 - indicator for r_in; set to 'rmax' for r_in=rmax or 'r0' for r_in=r0
%   fcor [s-1] - Coriolis parameter (only used for sign of Vazim)
%   CkCd [-] - ratio of surface exchange coefficients (Ck/Cd)
%   rr_ER11 [m] - vector of radii at which to calculate ER11 solution
%
% Outputs:
%   V_ER11 [ms-1] - vector of wind speeds of ER11 solution
%   r_out [m] - opposite of r_in (rmax vs. r0), taken directly from profile
%
% Example: 
%   [V_ER11,r_out] = ER11_radprof(Vmax_ER11,r0_ER11,'r0',fcor,CkCd,rr_mean)
%
% Reference: Emanuel, K., and R. Rotunno, 2011: Self-Stratification of
%   Tropical Cyclone Outflow. Part I: Implications for Storm Structure.
%   J. Atmos. Sci., 68, 2236-2249.
%
% Other m-files required: ER11_radprof_raw
% Subfunctions: none
% MAT-files required: none
%
% See also:

% Author: Dan Chavas
% CEE Dept, Princeton University
% email: drchavas@gmail.com
% Website: --
% 10 Dec 2013; Last revision: 20 Oct 2014

% Revisions:
% 3 Mar 2014 - Updated r0 interpolation algorithm to cut out r<rmax
% 20 Oct 2014 - set first argument of first while loop to ">" instead of
%   ">="; otherwise rmax values at exactly dr/2 intervals (e.g. 10.5 for
%   dr=1 km) will not converge
%------------- BEGIN CODE --------------


function [V_ER11,r_out] = ER11_radprof(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11)

dr = rr_ER11(2)-rr_ER11(1);

%% Call ER11_radprof_raw
[V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11);

%% Calculate error in r_in
switch rmax_or_r0
    case 'rmax'
        drin_temp = r_in-rr_ER11(V_ER11==max(V_ER11));
    case 'r0'
        drin_temp = r_in-interp1(V_ER11(3:end),rr_ER11(3:end),0,'pchip');
end

%% Calculate error in Vmax
dVmax_temp = Vmax - max(V_ER11);

%% Check is errors are too large and adjust accordingly
r_in_save = r_in;
Vmax_save = Vmax;

%%NOTE: rmax and Vmax depend on one another!

hold off

%%r_in first
n_iter = 0;
while(abs(drin_temp)>dr/2 || abs(dVmax_temp/Vmax_save)>=10^-2) %if error is sufficiently large; NOTE: FIRST ARGUMENT MUST BE ">" NOT ">=" or else rmax values at exactly dr/2 intervals (e.g. 10.5 for dr=1 km) will not converge

    %drin_temp/1000
    
    n_iter = n_iter + 1;
    if(n_iter>20)
        %sprintf('ER11 CALCULATION DID NOT CONVERGE TO INPUT (RMAX,VMAX) = (%3.1f km,%3.1f m/s); Ck/Cd = %2.2f!',r_in_save/1000,Vmax_save,CkCd)
        V_ER11 = NaN(size(rr_ER11));
        r_out = NaN;
        break
    end
        
%       drrr = drin_temp/1000
%       dVmax_temp
    
    %%Adjust estimate of r_in according to error
    r_in = r_in + drin_temp;
    
    %%Vmax second
    while(abs(dVmax_temp/Vmax)>=10^-2) %if error is sufficiently large
    
%          dVmax_temp

        %%Adjust estimate of Vmax according to error
        Vmax = Vmax + dVmax_temp;

        [V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11);
        Vmax_prof = max(V_ER11);
        dVmax_temp = Vmax_save-Vmax_prof;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%TESTING: Plot ER11 profile %%%%%%%%
        %{
        figure(2354)
        plot(rr_ER11(V_ER11>=0)/1000,V_ER11(V_ER11>=0),'k')
        hold on
        xlabel('r [km]')
        ylabel('V_g [m/s]')
        grid on
        box on
        axis([0 500 0 110])
        %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    [V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11);
    Vmax_prof = max(V_ER11);
    dVmax_temp = Vmax_save-Vmax_prof;
    switch rmax_or_r0
        case 'rmax'
            drin_temp = r_in_save-rr_ER11(V_ER11==Vmax_prof);
        case 'r0'
            drin_temp = r_in_save-interp1(V_ER11(3:end),rr_ER11(3:end),0,'pchip');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%TESTING: Plot ER11 profile %%%%%%%%
    %{
    figure(2354)
    plot(rr_ER11(V_ER11>=0)/1000,V_ER11(V_ER11>=0),'k')
    hold on
    plot(r_in/1000,Vmax,'rx')
    xlabel('r [km]')
    ylabel('V_g [m/s]')
    grid on
    box on
    axis([0 500 0 110])
    'hi'
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% Vmax_prof = max(V_ER11)
% rmax_prof = rr_ER11(V_ER11==Vmax_prof)

%{
%% EYE: apply solid body rotation
Vmax_prof = max(V_ER11);
rmax_prof = rr_ER11(V_ER11 == Vmax_prof);
ii_eye = rr_ER11<rmax_prof;
rr_eye = rr_ER11(ii_eye);
V_eye = Vmax_prof*(rr_eye/rmax_prof);
V_ER11(ii_eye) = V_eye;
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%TESTING: Plot ER11 profile %%%%%%%%
%{
plot(rr_ER11/1000,V_ER11,'Color',[.7 0 0],'LineWidth',3)
xlabel('r [km]')
ylabel('V_g [m/s]')
grid on
box on
axis([0 r0/1000 0 1.1*Vmax])

%% Save plot
% cd(dir_home2)
% plot_filename = sprintf('ER11_radprof_r0input.pdf');
% saveas(gcf,plot_filename,'pdf')
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------- END OF CODE --------------

