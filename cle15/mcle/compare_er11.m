%% compare_er11.m
% Print E04 and ER11 curve values at specific r/r0 points so Python can
% compare them directly for the reference case (50 m/s, 800 km, 5e-5).

addpath(genpath(fileparts(mfilename('fullpath'))));

Vmax = 50.0; r0 = 800e3; fcor = 5e-5;
Cdvary = 1; C_d = 0.0015; w_cool = 0.002; CkCd = 1.0;
Nr = 200000;

%% E04 profile
[rrfracr0_E04, MMfracM0_E04] = E04_outerwind_r0input_nondim_MM0(r0, fcor, Cdvary, C_d, w_cool, Nr);
M0 = 0.5 * fcor * r0^2;

fprintf('=== E04 ===\n');
fprintf('N=%d  drfracr0=%.6f\n', numel(rrfracr0_E04), rrfracr0_E04(2)-rrfracr0_E04(1));
% Print 10 evenly-spaced samples
idx = round(linspace(1, numel(rrfracr0_E04), 10));
for k = idx
    fprintf('  r/r0=%.6f  M/M0=%.8f\n', rrfracr0_E04(k), MMfracM0_E04(k));
end

%% Trace E04 integration steps and print values at check points
fprintf('\n=== E04 integration trace ===\n');
Nr_local = 1000;
drfracr0 = 0.001;
rfracr0_max = 1.0;
rfracr0_min = rfracr0_max - (Nr_local-1)*drfracr0;
rrfracr0_local = rfracr0_min:drfracr0:rfracr0_max;
MMfracM0_local = NaN(size(rrfracr0_local));
MMfracM0_local(end) = 1.0;
MMfracM0_local(end-1) = 1.0;

C_d_lowV = 6.2e-4; V_thresh1 = 6; V_thresh2 = 35.4; C_d_highV = 2.35e-3;
linear_slope_loc = (C_d_highV-C_d_lowV)/(V_thresh2-V_thresh1);

rfracr0_temp = rrfracr0_local(end-1);
MfracM0_temp = MMfracM0_local(end);

% First 5 steps
for step = 1:5
    V_temp = (M0/r0)*((MfracM0_temp/rfracr0_temp) - rfracr0_temp);
    V_temp = max(0, V_temp);
    if V_temp <= V_thresh1; C_d_loc = C_d_lowV;
    elseif V_temp > V_thresh2; C_d_loc = C_d_highV;
    else; C_d_loc = C_d_lowV + linear_slope_loc*(V_temp-V_thresh1); end
    gamma_loc = C_d_loc * fcor * r0 / w_cool;
    M_term = MfracM0_temp - rfracr0_temp^2;
    denom = 1 - rfracr0_temp^2;
    dMdr = gamma_loc * M_term^2 / denom;
    M_new = MfracM0_temp - dMdr*drfracr0;
    save_idx = Nr_local - step - 1;  % 1-based: end-ii-1 with ii=step
    fprintf('  step %d: r/r0=%.6f  V=%.4f  C_d=%.4e  gamma=%.4f  dMdr=%.6f  M_new=%.8f  save_idx=%d (r/r0=%.6f)\n', ...
        step, rfracr0_temp, V_temp, C_d_loc, gamma_loc, dMdr, M_new, save_idx, rrfracr0_local(save_idx));
    MMfracM0_local(save_idx) = M_new;
    MfracM0_temp = M_new;
    rfracr0_temp = rfracr0_temp - drfracr0;
end

% Full E04 from the official function
fprintf('\n=== Official E04 at check points ===\n');
check_pts = [0.001, 0.112, 0.223, 0.334, 0.445, 0.556, 0.667, 0.778, 0.889, 1.000];
for pt = check_pts
    idx = round((pt - rfracr0_min)/drfracr0) + 1;  % 1-based
    if idx >= 1 && idx <= numel(rrfracr0_E04)
        fprintf('  r/r0=%.3f  E04_official=%.8f\n', rrfracr0_E04(idx), MMfracM0_E04(idx));
    end
end
