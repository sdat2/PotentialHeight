% take r0 as input, get central pressure

clear
clc
close all
addpath(genpath('mfiles/'));

% read in JSON file of inputs
fileName = 'data/inputs.json'; % filename in JSON extension
in_str = fileread(fileName); % dedicated for reading files as text
in = jsondecode(in_str); % Using the jsondecode function to parse JSON from string

% print inputs json in matlab
in.r0

%in_data("r0")

[rr, VV, rmax, rmerge, Vmerge] = ...
    ER11E04_nondim_r0input(in.Vmax, in.r0, in.fcor, in.Cdvary, in.Cd, in.w_cool, in.CkCdvary, in.CkCd, in.eye_adj, in.alpha_eye);

out.rr = rr;
out.VV = VV;
out.rmax = rmax;
out.rmerge = rmerge;
out.Vmerge = Vmerge;

% write out JSON file of outputs
out_str = jsonencode(out);
fid = fopen('data/outputs.json', 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, out_str, 'char');
fclose(fid);
