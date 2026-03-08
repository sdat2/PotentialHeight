%radprof_eyeadj.m -- Smooth adjustment to eye r/r_m < 1
%Purpose: adjust eye by mutliplying input data by (r/r_m)^alpha for r<r_m
%
% Syntax:  [VV_out] = radprof_eyeadj(rr_in,VV_in,alpha,r_eye_outer,V_eye_outer)
%
% Inputs:
%   rr_in [m] - radius vector
%   VV_in [m/s] - wind speeds at rr_in
%   alpha [] - eye wind profile exponent
%   r_eye_outer [m] - outer radius of eye modification (default = rmax)
%   V_eye_outer [m/s] - wind speed at r_eye_outer (default = Vmax)
%
% Outputs:
%   VV_out [ms-1] - wind speeds at rr_in with sbr applied
%
% Example: 
%   [VV_out] = radprof_eyeadj(rr_in,VV_in,alpha,r_eye_outer,V_eye_outer)
%
% Other m-files required:
% Subfunctions: none
% MAT-files required: none
%
% See also:

% Author: Dan Chavas
% CEE Dept, Princeton University
% email: drchavas@gmail.com
% Website: --
% 11 Sep 2014; Last revision:

% Revisions:
%------------- BEGIN CODE --------------


function [VV_out] = radprof_eyeadj(rr_in,VV_in,alpha,r_eye_outer,V_eye_outer)

switch nargin
    case 3
        V_eye_outer = max(VV_in); %default = Vmax in profile
        r_eye_outer = rr_in(VV_in==V_eye_outer); %default = rmax in profile
    case 2
        alpha = 1;  %default = 1
        V_eye_outer = max(VV_in); %default = Vmax in profile
        r_eye_outer = rr_in(VV_in==V_eye_outer); %default = rmax in profile

end

%% Start with input data
VV_out = VV_in;

%% Extract eye region
indices_eye = rr_in<=r_eye_outer;
rr_eye = rr_in(indices_eye);
VV_eye = VV_out(indices_eye);

%% Normalize r and V
rr_eye_norm = rr_eye./r_eye_outer;
VV_eye_norm = VV_eye./V_eye_outer;

%% Define multiplicative scaling factor
eye_factor = rr_eye_norm.^alpha;

%% Multiply data by scaling factor
VV_eye_norm_out = VV_eye_norm.*eye_factor;

%% Redimensionalize
VV_eye_out = V_eye_outer*VV_eye_norm_out;

%% Replace eye values
VV_out(indices_eye) = VV_eye_out;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%TESTING: Plot profile %%%%%%%%
%{
figure(83)
clf(83)
plot(rr_in/1000,VV_in,'Color','k')
hold on
plot(rr_in/1000,VV_out,'Color','b')
xlabel('r [km]')
ylabel('V [m/s]')
grid on
box on
axis([0 2*r_eye_outer/1000 0 1.1*max(VV_in)])
'hi'
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------- END OF CODE --------------

