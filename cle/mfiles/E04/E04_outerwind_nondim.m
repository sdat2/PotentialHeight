%E04_outerwind_nondim.m -- estimate r0 using Emanuel (2004) model
% Purpose: For a given set of input parameters and (R,V), return full
% Emanuel (2004) non-convecting outer wind profile out to r_0_E04
%
% Syntax: [r_E04,V_E04,r_0_E04,ruser_err] = ...
%   E04_outerwind(ruser,Vuser,fcor,Cdvary,C_d,w_cool,V_max)
%
%
%Inputs:
%    ruser - [m] radius of input wind speed
%    Vuser - [ms-1] input wind speed
%    fcor - [s-1] Coriolis parameter
%    Cdvary - [] 0: C_d constant; 1 : C_d=C_d(V) following Donelan et al (2004)
%    C_d - [] constant drag coefficient
%    w_cool - [ms-1] - radiative subsidence rate (positive = downwards)
%    V_max - [ms-1] integrate inwards to radius of this wind speed
%
%Outputs:
%   r_E04 - [m] vector of E04 radial profile radii
%   V_E04 - [ms-1] vector of E04 radial profile azimuthal wind speeds
%   r_0_E04 - [m] E04 model-estimated outer radius of vanishing wind, r_0_E04
%   ruser_err [ms-1] - 1 = error in r_E04(Vuser) vs. input ruser
%
% Example: 
%
% Other m-files required: E04_outerwind_r0input
% Subfunctions: none
% MAT-files required: none
%
% References:
% Emanuel, K., 2004:  Tropical Cyclone Energetics and Structure.
%   In Atmospheric Turbulence and Mesoscale Meteorology, E. Fedorovich, R.
%   Rotunno and B. Stevens, editors, Cambridge University Press, 280 pp.
% Chavas, D. R., and K. A. Emanuel (2010), A QuikSCAT climatology of
%   tropical cyclone size, Geophys. Res. Lett., 37, L18816
%
% All input and output data is in [m] and [m/s]
%
%% Performance notes
% 11 Oct 2013 (DRC): Estimate of r_0_E04 least accurate at VERY high values of
%   both input R and V for typical values of fcor, C_d, w_cool -- basically
%   the longer the distance between ruser and r_0_E04, the less well it performs.
%
% Author: Dan Chavas
% CEE Dept, Princeton University
% email: drchavas@gmail.com
% Website: --
% 11 Oct 2013; Last revision: 19 May 2014
% 
%%Revision history:
% 6 Dec 2013 - muted print to screen of n_iter and iter_precision
% 3 Jan 2014 - moved Testing plot inside of function 'END'; decreased
%   r_0_E04_update threshold to 10^-8 (even lower and it loops infinitely);
%   changed output r_0_E04_residual_error_flag to ruser_error (this is the
%   relevant error metric) and added warning for large relative ruser_error
% 19 May 2014 - updated error from Vuser to ruser, as this is guaranteed
%   to actually exist in each iteration of V_E04 guess
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------- BEGIN CODE --------------


function [rr,VV,r0,rrfracr0,MMfracM0] = E04_outerwind_nondim(ruser,Vuser,fcor,Cdvary,C_d,w_cool,Nr)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INTRO STUFF
switch nargin
    case 6
        Nr = 100000;    %[]; default to calculate entire profile
end
fcor = abs(fcor);
Muser = .5*fcor*ruser.^2 + ruser.*Vuser;

r0 = 2*ruser; %[m]; initial guess
r0_min = ruser;    %[m]; initial lower bound on r0
r0_max = 10000*1000;    %[m]; initial upper bound on r0
assert(r0<r0_max,'ruser is way too big')
MuserM0_err_thresh = .0001;   %[m]; minimum allowable error in ruser

%% Try initial r0 guess and return error in Vuser and update r0 accordingly
[rrfracr0_temp,MMfracM0_temp] = E04_outerwind_r0input_nondim_MM0(r0,fcor,Cdvary,C_d,w_cool,Nr);
M0 = .5*fcor*r0^2;
MuserM0 = Muser/M0;
ruserr0 = ruser/r0;
MuserM0_temp = interp1(rrfracr0_temp,MMfracM0_temp,ruserr0,'pchip',NaN); %must cut off r0 point since M constant at final two points

%%Initial error
MuserM0_err = MuserM0_temp - MuserM0;  %error in guess

%%Initial dr0
if(MuserM0_err>0)    %guess is too high
    dr0 = abs(r0-r0_min);   %[m]
else                %guess is too low
    dr0 = abs(r0_max-r0);   %[m]
end

%% TESTING %%%%%%%%%%%%%%%%%%%%
%{
figure(912)
clf(912)
plot(rrfracr0_temp,MMfracM0_temp,'b')
hold on
plot(ruserr0,MuserM0,'r*','MarkerSize',20)
plot(ruserr0,MuserM0_temp,'b*','MarkerSize',20)
'hi'
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Converge geometrically (halving/doubling) on correct r0
while(abs(MuserM0_err)>MuserM0_err_thresh)

    %%Update to new values
    dr0 = dr0/2;    %[m]
    if(MuserM0_err>0)    %guess is too high
        r0 = r0 - dr0;
    else                %guess is too low
        r0 = r0 + dr0;
    end
    
    [rrfracr0_temp,MMfracM0_temp] = E04_outerwind_r0input_nondim_MM0(r0,fcor,Cdvary,C_d,w_cool,Nr);
    M0 = .5*fcor*r0^2;
    MuserM0 = Muser/M0;
    ruserr0 = ruser/r0;
    MuserM0_temp = interp1(rrfracr0_temp,MMfracM0_temp,ruserr0,'pchip',NaN); %must cut off r0 point since M constant at final two points

    MuserM0_err = MuserM0_temp - MuserM0;  %error in guess
    
    
    %% TESTING %%%%%%%%%%%%%%%%%%%%
    %{
    figure(912)
    plot(rrfracr0_temp,MMfracM0_temp,'b')
    hold on
    plot(ruserr0,MuserM0,'r*','MarkerSize',20)
    plot(ruserr0,MuserM0_temp,'b*','MarkerSize',20)
    'hi'
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

%% Save final data
rrfracr0 = rrfracr0_temp;
MMfracM0 = MMfracM0_temp;

%% Calculate dimensional wind speed and radii
VV = (M0/r0)*((MMfracM0./rrfracr0)-rrfracr0);  %[ms-1]
rr = rrfracr0*r0;   %[m]


end
%------------- END OF CODE --------------
