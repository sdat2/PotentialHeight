% take r0 as input, get central pressure
% octave --no-gui --no-gui-libs r0_pm.m
% octave --no-gui --no-gui-libs r0_pm.m test
clear
clc
% warning("off", "Octave:possible-matlab-short-circuit-operator")
warning("off", "all")

# Check if an argument was provided
args = argv();
if numel(args) < 1
  name = "";
else
  name = args{1};
end

% close all
% add path to mcle and mfiles
current_folder = pwd;
file_path = mfilename('fullpath');
% take away the file name
file_path = fileparts(file_path);
file_path_cle = fileparts(file_path);

addpath(genpath(file_path)); % 'mcle/'));
file_path = char([file_path, "/", "mfiles"]);
addpath(genpath(char([file_path, "/", "mfiles"])));  % 'mcle/mfiles/'));

% read in JSON file of inputs
if isempty(name)
  data_folder = char([file_path_cle, "/", "data"]);
  fileName = char([data_folder, '/', 'inputs.json']); % filename in JSON extension
else
  data_folder = char([file_path_cle, "/", "data", "/", "tmp"]);
  fileName = char([data_folder, '/', name, '-inputs.json']);
  % filename in JSON extension
end

fprintf('Attempting to read file: %s\n', fileName);
if exist(fileName, 'file')
  fprintf('File %s exists\n', fileName);
else
  warningMessage = sprintf('Warning: file does not exist:\n%s', FileName);
  uiwait(msgbox(warningMessage));
end

in_str = fileread(fileName); % dedicated for reading files as text
inputs = jsondecode(in_str); % Using the jsondecode function to parse JSON from string

% print inputs json in matlab
inputs.r0;

[rr, VV, rmax, rmerge, Vmerge] = ...
    ER11E04_nondim_r0input(inputs.Vmax, inputs.r0, inputs.fcor, inputs.Cdvary, inputs.Cd, inputs.w_cool, inputs.CkCdvary, inputs.CkCd, inputs.eye_adj, inputs.alpha_eye);

out.rr = rr;
out.VV = VV;
out.rmax = rmax;
out.rmerge = rmerge;
out.Vmerge = Vmerge;

% write out JSON file of outputs
out_str = jsonencode(out);

if isempty(name)
  fid = fopen(char([data_folder, '/', 'outputs.json']), 'w');
else
  fid = fopen(char([data_folder, '/' , name, '-outputs.json']), 'w');
end

if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, out_str, 'char');
fclose(fid);

