%ER11_radprof_raw_nondim.m -- ER11 profile, calculated non-dimensionally
%
% Syntax:  [rr,VV,rmax,rrfracrm,MMfracMm] = ER11_radprof_raw_nondim(Vmax,r_in,rmax_or_r0,fcor,CkCd)
%
% Inputs:
%   Vmax [ms-1] - maximum wind speed
%   r_in [m] - either radius of maximum wind (rmax) or outer radius of
%      vanishing wind (r0)
%   rmax_or_r0 - indicator for r_in; set to 'rmax' for r_in=rmax or 'r0' for r_in=r0
%   fcor [s-1] - Coriolis parameter
%   CkCd [-] - ratio of surface exchange coefficients (Ck/Cd)
%
% Outputs:
%   rr [m] - vector of radii for entire profile
%   VV [ms-1] - vector of wind speeds at rr
%   rmax [m] - radius of maximum wind in profile (rr)
%   Vmax [m] - maximum wind in profile (VV)
%   rrfracrm [-] - vector of r/rmax
%   MMfracMm [-] - vector of M/Mmax at rrfracrm
%
% Example: 
%   [rr,VV,rmax,rrfracrm,MMfracMm] = ER11_radprof_raw_nondim(Vmax,r_in,rmax_or_r0,fcor,CkCd)
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
% 2015-05-11; Last revision:
%------------- BEGIN CODE --------------


function [rr,VV,rmax,Vmax,rrfracrm,MMfracMm] = ER11_radprof_raw_nondim(Vmax,r_in,rmax_or_r0,fcor,CkCd)

fcor = abs(fcor);

switch rmax_or_r0
    case 'r0'
        r0 = r_in;
        
        %%Full solution for rmax-r0 relationship (ER11 Eq. 37 -- with M_max including fcor term!)
        syms rmax_var
        s_temp = solve( ((.5*fcor*r0^2)/(Vmax*rmax_var + .5*fcor*rmax_var^2))^(2-CkCd) == ...
            (2*(r0/rmax_var)^2)/(2-CkCd+CkCd*(r0/rmax_var)^2), rmax_var);
        s_temp = eval(s_temp);
        rmax = s_temp(s_temp<r0 & s_temp>0);
                
    case 'rmax'
        rmax = r_in;

    otherwise
        assert('rmax_or_r0 must be set to either "r0" or "rmax"')
end

%% CALCULATE Emanuel and Rotunno (2011) theoretical profile
Mm = .5*fcor*rmax.^2 + rmax.*Vmax; %[m2/s]; M at rmax
drfracrm = .01;
if(rmax>100*1000)
    drfracrm = drfracrm/10; %extra precision for large storm
end
rfracrm_min = 0;   %[-]; start at r=0
rfracrm_max = 50;    %[-]; extend out to many rmaxs
rrfracrm = rfracrm_min:drfracrm:rfracrm_max; %[]; r/r0 vector
MMfracMm = (2*rrfracrm.^2)./(2-CkCd+CkCd*rrfracrm.^2).^(1/(2-CkCd));

%% Calculate dimensional wind speed and radii
rr = rrfracrm*rmax;   %[m]
VV = (Mm/rmax)*MMfracMm./rrfracrm - .5*fcor*rr;  %[ms-1]

%% Keep good data
ii_good = VV>=0;
rrfracrm = rrfracrm(ii_good);
MMfracMm = MMfracMm(ii_good);
rr = rr(ii_good);
VV = VV(ii_good);
Vmax = max(VV);
rmax = rr(VV==Vmax);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%TESTING: Plot ER11 profile %%%%%%%%
%{
figure(32)
clf(32)
plot(rrfracrm,MMfracMm,'Color','b')
hold on
plot(1,1,'rx')
xlabel('r/r_m')
ylabel('M/Mm')

figure(33)
clf(33)
plot(rr/1000,VV,'Color','b')
hold on
plot(rmax/1000,Vmax,'rx')
xlabel('r [km]')
ylabel('V [m/s]')
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
%------------- END OF CODE --------------

