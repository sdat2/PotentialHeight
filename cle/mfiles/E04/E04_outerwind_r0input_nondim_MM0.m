%E04_outerwind_r0input_nondim_noV.m -- Emanuel (2004) model calculation
% Purpose: Emanuel (2004) non-convecting outer wind profile, given r0
%
% Syntax: [rrfracr0,MMfracM0] = E04_outerwind_r0input_nondim_noV(r0,fcor,Cdvary,C_d,w_cool,Nr);
%
%
%Inputs:
%    r0 - [m] outer radius where V=0
%    fcor - [s-1] Coriolis parameter
%    Cdvary - [] 0: C_d constant; 1 : C_d=C_d(V) following Donelan et al (2004)
%    C_d - [] drag coefficient; ignored if Cdvary = 1
%    w_cool - [ms-1] radiative subsidence rate (positive = downwards)
%    Nr - [-] number of radial nodes inwards of r0; default will calculate
%       entire profile
%
%Outputs:
%   rrfracr0 - [-] vector of r/r0
%   MMfracM0 - [-] vector of M/M0 at rrfracr0
%
% Example: 
%   [rrfracr0,MMfracM0] = E04_outerwind_r0input_nondim_noV(r0,fcor,Cdvary,C_d,w_cool,Nr);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
% Emanuel, K., 2004:  Tropical Cyclone Energetics and Structure.
%   In Atmospheric Turbulence and Mesoscale Meteorology, E. Fedorovich, R.
%   Rotunno and B. Stevens, editors, Cambridge University Press, 280 pp.
% Chavas, D.R. and N. Lin 2015.
%
% All input and output data is in [m] and [m/s]
%
% Author: Dan Chavas
% CEE Dept, Princeton University
% email: drchavas@gmail.com
% Website: --
% 11 May 2015; Last revision:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------- BEGIN CODE --------------

function [rrfracr0,MMfracM0] = E04_outerwind_r0input_nondim_MM0(r0,fcor,Cdvary,C_d,w_cool,Nr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
fcor = abs(fcor);
M0 = .5*fcor*r0.^2; %[m2/s]; M at outer radius

drfracr0 = .001;
if(r0>2500*1000 | r0<200*1000)
    drfracr0 = drfracr0/10; %extra precision for very large storm to avoid funny bumps near r0 (though rest of solution is stable!)
end                         %or for tiny storm that requires E04 extend to very small radii to match with ER11

switch nargin
    case 5
        Nr = 1/drfracr0;    %[]; default number of radial nodes
end

if(Nr > 1/drfracr0)
    Nr = 1/drfracr0;    %grid radii must be > 0
end

rfracr0_max = 1;   %[-]; start at r0, move radially inwards
rfracr0_min = rfracr0_max - (Nr-1)*drfracr0;    %[-]; inner-most node
rrfracr0 = rfracr0_min:drfracr0:rfracr0_max; %[]; r/r0 vector
MMfracM0 = NaN(size(rrfracr0));  %[]; M/M0 vector initialized to 1 (M/M0 = 1 at r/r0=1)
MMfracM0(end) = 1;

%% First step inwards from r0: d(M/M0)/d(r/r0) = 0 by definition
rfracr0_temp = rrfracr0(end-1); %one step inwards from r0
%dMfracM0_drfracr0_temp = 0;  %[]; d(M/M0)/d(r/r0) = 0 at r/r0 = 1;
MfracM0_temp = MMfracM0(end);
MMfracM0(end-1) = MfracM0_temp;

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variable C_d: code from Cd_Donelan04.m (function call is slow) %%%%%%
%%Piecewise linear fit parameters estimated from Donelan2004_fit.m
C_d_lowV = 6.2e-4;
V_thresh1 = 6;  %m/s; transition from constant to linear increasing
V_thresh2 = 35.4;  %m/s; transition from linear increasing to constant
C_d_highV = 2.35e-3;
linear_slope = (C_d_highV-C_d_lowV)/(V_thresh2-V_thresh1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Integrate inwards from r0 to obtain profile of M/M0 vs. r/r0
for ii=1:Nr-2   %first two nodes already done above
                
    %% Calculate C_d varying with V, if desired
    if(Cdvary==1)

        %%Calculate V at this r/r0 (for variable C_d only)
        V_temp = (M0/r0)*((MfracM0_temp./rfracr0_temp)-rfracr0_temp);

        %%Calculate C_d
        if(V_temp<=V_thresh1)
            C_d = C_d_lowV;
        elseif(V_temp>V_thresh2)
            C_d = C_d_highV;
        else
            C_d = C_d_lowV + linear_slope*(V_temp-V_thresh1);
        end
        
    end
        
    %% Calculate model parameter, gamma
    gam = C_d*fcor*r0/w_cool;   %[]; non-dimensional model parameter

    %% Update dMfracM0_drfracr0 at next step inwards
    dMfracM0_drfracr0_temp = gam*((MfracM0_temp-rfracr0_temp.^2).^2)/(1-rfracr0_temp.^2);
    
    %% Integrate M/M0 radially inwards
    MfracM0_temp = MfracM0_temp - dMfracM0_drfracr0_temp*drfracr0;
    
    %% Update r/r0 to follow M/M0
    rfracr0_temp = rfracr0_temp - drfracr0; %[] move one step inwards

    %% Save updated values
    MMfracM0(end-ii-1) = MfracM0_temp;

        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TESTING: M/M0 vs. r/r0 %%%%%%
%{
figure(98)
hold off
plot(rrfracr0,MMfracM0,'b','LineWidth',2)
hold on
plot(1,1,'r*','MarkerSize',14,'LineWidth',2)
xlabel('r/r_0')
ylabel('M/M_0')
input_title = sprintf('r_0 = %5.0f [km]',r0/1000);
title(input_title)
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TESTING: Make plot of radial profile %%%%%%
%{
figure(99)
hold off
plot(rr/1000,VV,'b','LineWidth',2)
hold on
plot(r0/1000,0,'r*','MarkerSize',14,'LineWidth',2)
xlabel('r [km]')
ylabel('V [m/s]')
input_title = sprintf('r_0 = %5.0f [km]',r0/1000);
title(input_title)
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
%------------- END OF CODE --------------
