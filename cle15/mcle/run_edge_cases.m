%% run_edge_cases.m
% Run ER11E04_nondim_r0input on the physically degenerate edge cases and
% print rmax so we can compare directly with the Python / numba solvers.

addpath(genpath(fileparts(mfilename('fullpath'))));

% Default physical parameters (matching Python defaults in w22/constants.py)
Cdvary   = 0;       % constant C_d (matches CDVARY_DEFAULT=0 in Python)
C_d      = 0.0015;  % drag coefficient (matches CD_DEFAULT)
w_cool   = 0.002;   % m/s (matches W_COOL_DEFAULT)
CkCdvary = 0;       % fixed CkCd (matches CKCDVARY_DEFAULT=0)
CkCd     = 0.9;     % matches CK_CD_DEFAULT (standard PI value)
eye_adj  = 0;       % matches EYE_ADJ_DEFAULT=0
alpha_eye = 0.5;    % matches ALPHA_EYE_DEFAULT=0.5

edge_cases = {
    90.0,  200e3, 3e-5, 'high Vmax small r0 low f';
    90.0,  200e3, 5e-5, 'high Vmax small r0 mid f';
    90.0,  200e3, 7e-5, 'high Vmax small r0 high f';
    20.0,  200e3, 3e-5, 'low Vmax  small r0 low f';
    90.0, 2000e3, 3e-5, 'high Vmax large r0 low f';
    20.0, 2000e3, 3e-5, 'low Vmax  large r0 low f';
    50.0,  800e3, 5e-5, 'mid Vmax  mid r0  mid f (reference)';
};

fprintf('%-45s  %12s  %12s\n', 'case', 'rmax (km)', 'rmaxr0');
fprintf('%s\n', repmat('-', 1, 75));

for k = 1:size(edge_cases, 1)
    Vmax = edge_cases{k,1};
    r0   = edge_cases{k,2};
    fcor = edge_cases{k,3};
    label = edge_cases{k,4};

    try
        [~, ~, rmax, ~, ~, ~, ~, rmaxr0] = ...
            ER11E04_nondim_r0input(Vmax, r0, fcor, ...
                Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye);
        fprintf('%-45s  %12.4f  %12.6f\n', label, rmax/1e3, rmaxr0);
    catch e
        fprintf('%-45s  ERROR: %s\n', label, e.message);
    end
end
