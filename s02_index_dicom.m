%% ========================================================================
%  s02_index_dicom.m -- Build a table of every usable MRI series.
%  RUN:    press the green Run button
%  TIME:   10-20 minutes (reads one header per series folder)
%  OUTPUT: study\series_index.mat
%% ========================================================================
clear; clc;
cfg = s00_config();

patients = dir(fullfile(cfg.dicomRoot,'ProstateX-*'));
patients = patients([patients.isdir]);
fprintf('Found %d patient folders.\n\n', numel(patients));
if isempty(patients)
    error('No patients found. Check cfg.dicomRoot in s00_config.m');
end

rows = {};
t0 = tic;

for p = 1:numel(patients)
    pid  = patients(p).name;
    pdir = fullfile(patients(p).folder, pid);

    dcm = dir(fullfile(pdir,'**','*.dcm'));
    if isempty(dcm), continue; end
    folders = unique({dcm.folder});        % each folder = one series

    for s = 1:numel(folders)
        f = dir(fullfile(folders{s},'*.dcm'));
        if isempty(f), continue; end
        try
            info = dicominfo(fullfile(f(1).folder, f(1).name));
        catch
            continue;                       % unreadable -> skip
        end

        desc = '';
        if isfield(info,'SeriesDescription'); desc = info.SeriesDescription; end

        key = localClassify(desc, cfg);     % which of the 3 sequences is it?
        if isempty(key), continue; end      % not one we want -> skip

        num = NaN;
        if isfield(info,'SeriesNumber'); num = double(info.SeriesNumber); end
        mdl = '';
        if isfield(info,'ManufacturerModelName'); mdl = info.ManufacturerModelName; end

        rows(end+1,:) = {pid, key, string(desc), num, ...
                         numel(f), string(mdl), folders{s}};   %#ok<SAGROW>
    end

    if mod(p,25)==0
        fprintf('  %3d/%d patients   %5d series kept   %.1f min elapsed\n', ...
                p, numel(patients), size(rows,1), toc(t0)/60);
    end
end

S = cell2table(rows,'VariableNames', ...
    {'ProxID','Sequence','SeriesDesc','SeriesNumber','nFiles','Scanner','Folder'});
save(fullfile(cfg.work,'series_index.mat'),'S');

%% ------------------------------ REPORT ---------------------------------
fprintf('\n================= RESULT =================\n');
fprintf('Usable series kept : %d\n', height(S));
fprintf('Patients with data : %d\n\n', numel(unique(S.ProxID)));

fprintf('Series per sequence:\n');
for q = {'t2','adc','dwi'}
    n  = sum(strcmp(S.Sequence,q{1}));
    np = numel(unique(S.ProxID(strcmp(S.Sequence,q{1}))));
    fprintf('   %-4s  %4d series   %3d patients\n', q{1}, n, np);
end

up = unique(S.ProxID);
complete = 0;
for k = 1:numel(up)
    sq = unique(S.Sequence(strcmp(S.ProxID,up{k})));
    if numel(sq)==3, complete = complete + 1; end
end
fprintf('\nPatients with ALL THREE sequences: %d   <-- these are usable\n', complete);

fprintf('\nScanners:\n');
disp(groupcounts(S,'Scanner'));
fprintf('Total time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');

%% ---------------------- helper function --------------------------------
function key = localClassify(desc, cfg)
% Decide which sequence this series is. Returns '' if we don't want it.
key = '';
d = lower(string(desc));

for e = cfg.patExclude                 % reject unwanted series first
    if contains(d, lower(e{1})), return; end
end

for f = {'adc','dwi','t2'}             % ORDER MATTERS: adc, then dwi, then t2
    for q = 1:numel(cfg.pat.(f{1}))
        if contains(d, lower(cfg.pat.(f{1}){q}))
            key = f{1}; return;
        end
    end
end
end