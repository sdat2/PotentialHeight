% take r0 as input, get central pressure
% octave --no-gui --no-gui-libs r0_pm.m
% octave --no-gui --no-gui-libs r0_pm.m ~/data_folder test
clear
clc
% warning("off", "Octave:possible-matlab-short-circuit-operator")
warning("off", "all")


% close all
% add path to mcle and mfiles
current_folder = pwd;
file_path = mfilename('fullpath');
% take away the file name
file_path = fileparts(file_path);
file_path_cle = fileparts(file_path);

addpath(genpath(file_path));
file_path = char([file_path, "/", "mfiles"]);
addpath(genpath(char([file_path, "/", "mfiles"])));


# Check if an argument was provided
args = argv();
if numel(args) < 2
  name = "";
  data_folder = char([file_path_cle, "/", "data"]);
else
  data_folder = args{1};
  name = args{2};
end

% read in JSON file of inputs
if isempty(name)
  data_folder = char([file_path_cle, "/", "data"]);
  input_file_name = char([data_folder, '/', 'inputs.json']); % filename in JSON extension
else
  % data_folder = char([file_path_cle, "/", "data", "/", "tmp"]);
  input_file_name = char([data_folder, '/', name, '-inputs.json']);
  % filename in JSON extension
end

fprintf('Attempting to read file: %s\n', input_file_name);
if exist(input_file_name, 'file')
  fprintf('File %s exists\n', input_file_name);
else
  fprintf('File %s does not exist\n', input_file_name);
  warningMessage = sprintf('Warning: file does not exist:\n%s', input_file_name);
  % uiwait(msgbox(warningMessage));
end

in_str = fileread(input_file_name); % dedicated for reading files as text
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
  output_file_name = char([data_folder, '/', 'outputs.json'])
else
  output_file_name = char([data_folder, '/' , name, '-outputs.json'])
end

fprintf('Attempting to write file: %s\n', output_file_name);

fid = fopen(output_file_name, 'w');

if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, out_str, 'char');
fclose(fid);


if exist(output_file_name, 'file')
  fprintf('File %s exists\n', output_file_name);
else
  fprintf('File %s does not exist\n', output_file_name);
  warningMessage = sprintf('Warning: file does not exist:\n%s', output_file_name);
  % uiwait(msgbox(warningMessage));
end
