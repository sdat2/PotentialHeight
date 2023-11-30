%ER11_radprof_nondim.m -- ER11_radprof converged on correct (Vmax,r_in)
%
% Syntax:  [VV,r_out] = ER11_radprof(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr)
%
% Inputs:
%   Vmax [ms-1] - maximum wind speed
%   r_in [m] - either radius of maximum wind (rmax) or outer radius of
%      vanishing wind (r0)
%   rmax_or_r0 - indicator for r_in; set to 'rmax' for r_in=rmax or 'r0' for r_in=r0
%   fcor [s-1] - Coriolis parameter (only used for sign of Vazim)
%   CkCd [-] - ratio of surface exchange coefficients (Ck/Cd)
%   rr [m] - vector of radii at which to calculate ER11 solution
%
% Outputs:
%   VV [ms-1] - vector of wind speeds of ER11 solution
%   r_out [m] - opposite of r_in (rmax vs. r0), taken directly from profile
%
% Example: 
%   [VV,r_out] = ER11_radprof(Vmax_ER11,r0_ER11,'r0',fcor,CkCd,rr_mean)
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


function [rr,VV,rrfracrm,MMfracMm] = ER11_radprof_nondim(Vmax_data,rmax_data,fcor,CkCd)

%% First try setting rmax = rmax_data
rmax_or_r0 = 'rmax';
[rr,VV,~,~,~] = ER11_radprof_raw_nondim(Vmax_data,rmax_data,rmax_or_r0,fcor,CkCd);

%% Calculate error in rmax, Vmax
Vmax_prof = max(VV);
rmax_prof = rr(VV==Vmax_prof);
drmax_temp = rmax_data - rmax_prof;
dVmax_temp = Vmax_data - Vmax_prof;

%% Check if errors are too large and adjust accordingly
%%NOTE: rmax and Vmax depend on one another!

%%rmax first
n_iter = 0;
rmax = rmax_data;   %initialize with data value
Vmax = Vmax_data;   %initialize with data value
drmax_temp_old = 10^9;  %initialize to something large
while(abs(drmax_temp/rmax_data)>10^-2 || abs(dVmax_temp/Vmax_data)>=10^-2)
    
    n_iter = n_iter + 1;
    if(n_iter>200)
        sprintf('ER11 CALCULATION DID NOT CONVERGE TO INPUT (RMAX,VMAX) = (%3.1f km,%3.1f m/s); Ck/Cd = %2.2f!',rmax_data/1000,Vmax_data,CkCd)
        VV = NaN(size(rr));
        break
    end
            
    %%Adjust estimate of rmax according to error
    rmax = rmax + drmax_temp;
    
    %%Vmax second
    while(abs(dVmax_temp/Vmax)>=10^-2) %if error is sufficiently large
    
%          dVmax_temp

        %%Adjust estimate of Vmax according to error
        Vmax = Vmax + dVmax_temp;

        [~,VV,~,~,~] = ER11_radprof_raw_nondim(Vmax,rmax,rmax_or_r0,fcor,CkCd);
        Vmax_prof = max(VV);
        dVmax_temp = Vmax_data-Vmax_prof;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%TESTING: Plot ER11 profile %%%%%%%%
        %{
        figure(2354)
        plot(rr(VV>=0)/1000,VV(VV>=0),'k')
        hold on
        xlabel('r [km]')
        ylabel('V_g [m/s]')
        grid on
        box on
        axis([0 500 0 110])
        %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    %%Save old data
    rr_old = rr;
    VV_old = VV;
    drmax_temp_old = drmax_temp;    %save old version
    
    [rr,VV,~,~,~] = ER11_radprof_raw_nondim(Vmax,rmax,rmax_or_r0,fcor,CkCd);
    
    %% Calculate error in rmax, Vmax
    Vmax_prof = max(VV);
    rmax_prof = rr(VV==Vmax_prof);
    drmax_temp = rmax_data - rmax_prof;
    dVmax_temp = Vmax_data - Vmax_prof;

    if(abs(drmax_temp)>abs(drmax_temp_old))   %starting to blow up
        sprintf('ER11 will blow up, using best solution; inputs = (%3.1f km,%3.1f m/s); Ck/Cd = %2.2f!',rmax_data/1000,Vmax_data,CkCd)
        rr = rr_old;
        VV = VV_old;
        break
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%TESTING: Plot ER11 profile %%%%%%%%
    %{
    figure(2354)
    plot(rr(VV>=0)/1000,VV(VV>=0),'k')
    hold on
    xlabel('r [km]')
    ylabel('V_g [m/s]')
    grid on
    box on
    axis([0 500 0 110])
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if(Vmax<0 || rmax<0)
        sprintf('ER11 CALCULATION FAILED FOR (RMAX,VMAX) = (%3.1f km,%3.1f m/s); Ck/Cd = %2.2f!',rmax_data/1000,Vmax_data,CkCd)
        VV = NaN(size(rr));
        break
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%TESTING: Plot ER11 profile %%%%%%%%
%{
figure(2355)
plot(rr(VV>=0)/1000,VV(VV>=0),'k')
hold on
plot(rmax_data/1000,Vmax_data,'rx')
xlabel('r [km]')
ylabel('V_g [m/s]')
grid on
box on
axis([0 500 0 110])
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Calculate M/Mm vs. r/rm -- (rmax,Vmax) are DATA values (also profile values, since they now match)
rrfracrm = rr/rmax_data;
Mm = .5*fcor*rmax_data.^2 + rmax_data.*Vmax_data;  %t
MMfracMm = (.5*fcor*rr.^2 + rr.*VV)./Mm;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%TESTING: Plot ER11 profile %%%%%%%%
%{
plot(rr/1000,VV,'Color',[.7 0 0],'LineWidth',3)
xlabel('r [km]')
ylabel('V_g [m/s]')
grid on
box on
axis([0 r0/1000 0 1.1*Vmax])
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------- END OF CODE --------------

