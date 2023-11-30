%ER11_radprof_raw.m -- calculate original ER11 radial wind profile
%Purpose: ER11 direct from original equation, with no convergence for rmax
%or r0
%
% Syntax:  [V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,fcor,CkCd,rr_ER11)
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
%   [V_ER11,r_out] = ER11_radprof_raw(Vmax_ER11,r0_ER11,'r0',fcor,CkCd,rr_mean)
%
% Reference: Emanuel, K., and R. Rotunno, 2011: Self-Stratification of
%   Tropical Cyclone Outflow. Part I: Implications for Storm Structure.
%   J. Atmos. Sci., 68, 2236-2249.
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also:

% Author: Dan Chavas
% CEE Dept, Princeton University
% email: drchavas@gmail.com
% Website: --
% 10 Dec 2013; Last revision: 10 Jul 2014

% Revisions:
% 3 Mar 2014 - Updated r0 interpolation algorithm to cut out r<rmax
% 10 Jul 2014 - Fixed error in M_max
%------------- BEGIN CODE --------------


function [V_ER11,r_out] = ER11_radprof_raw(Vmax,r_in,rmax_or_r0,fcor,CkCd,rr_ER11)

fcor = abs(fcor);

switch rmax_or_r0
    case 'r0'
        r0 = r_in;
        
        %%Simple: assume V>>fr (ER11 Eq. 38)
        rmax_simple = ((.5*fcor*r0^2)/Vmax)*((.5*CkCd)^(1/(2-CkCd)));
        
        %%Full solution for rmax-r0 relationship (ER11 Eq. 37 -- with M_max including fcor term!)
        syms rmax_var
        s_temp = solve( ((.5*fcor*r0^2)/(Vmax*rmax_var + .5*fcor*rmax_var^2))^(2-CkCd) == ...
            (2*(r0/rmax_var)^2)/(2-CkCd+CkCd*(r0/rmax_var)^2), rmax_var);
        s_temp = eval(s_temp);
        rmax = s_temp(s_temp<r0 & s_temp>0);
        
        r_out = rmax;
        
%         if(fcor>0)
%             assert(abs(rmax-rmax_simple)<.1*rmax_simple,'Large difference between rmax and rmax_simple')
%         end
        
    case 'rmax'
        rmax = r_in;
%{        
        %%Simple: assume V>>fr (ER11 Eq. 38)
        r0_simple = sqrt(((Vmax*rmax)/((.5*CkCd)^(1/(2-CkCd))))/(.5*fcor));
        
        %%Full solution for rmax-r0 relationship (ER11 Eq. 37 -- with M_max including fcor term!)
        syms r0_var
        s_temp = solve( ((.5*fcor*r0_var^2)/(Vmax*rmax + .5*fcor*rmax^2))^(2-CkCd) == ...
            (2*(r0_var/rmax)^2)/(2-CkCd+CkCd*(r0_var/rmax)^2), r0_var);
        s_temp = eval(s_temp);
        r0 = s_temp(find(s_temp>rmax,1));
        
        r_out = r0;
%}
%         if(fcor>0)
%             assert(abs(r0-r0_simple)<.1*r0_simple,'Large difference between r0 and r0_simple')
%         end
        
    otherwise
        assert('rmax_or_r0 must be set to either "r0" or "rmax"')
end

%% CALCULATE Emanuel and Rotunno (2011) theoretical profile
V_ER11 = (1./rr_ER11).*(Vmax*rmax + .5*fcor*rmax^2).*((2*(rr_ER11./rmax).^2)./(2-CkCd+CkCd*(rr_ER11./rmax).^2)).^(1/(2-CkCd)) - .5*fcor*rr_ER11;

%%make V=0 at r=0
V_ER11(rr_ER11==0) = 0;

%%Return opposite r_in from profile
switch rmax_or_r0
    case 'r0'
        rmax_profile = rr_ER11(V_ER11==max(max(V_ER11)));
        r_out = rmax_profile; %use value from profile itself

        %%TESTING: rmax_profile interpolation %%%%%%
        %{
        figure(993)
        plot(rr_ER11,V_ER11)
        hold on
        plot(rmax_profile,max(V_ER11),'rx')
        %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%Check that there is not a huge discrepancy between the two
%         rmax_error = abs((rmax_profile-rmax)/rmax_profile);
%         assert(rmax_error<10^-2,...
%             sprintf('rmax analytical calculation does not match profile by %3.1f [km]',1000*rmax_error))
        
    case 'rmax'
        i_rmax = find(V_ER11==max(V_ER11));
        r0_profile = interp1(V_ER11(i_rmax+1:end),rr_ER11(i_rmax+1:end),0,'pchip');
        r_out = r0_profile; %use value from profile itself

        %%TESTING: r0_profile interpolation %%%%%%
        %{
        figure(993)
        plot(rr_ER11,V_ER11)
        hold on
        plot(r0_profile,0,'rx')
        %}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%Check that there is not a huge discrepancy between the two
%         r0_error = abs((r0_profile-r0)/r0_profile);
%         assert(r0_error<10^-2,...
%             sprintf('r0 analytical calculation does not match profile by %3.1f [km]',1000*r0_error))
        
end

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

