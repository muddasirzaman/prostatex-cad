%% ========================================================================
%  s04_create_splits.m -- Patient-level stratified CV splits (5 folds x 5 seeds)
%  RUN:    press the green Run button
%  OUTPUT: study\splits\splits.mat   (splits + T_lesions)
%% ========================================================================
clear; clc;
cfg = s00_config();

% ---- 1. Read every extracted patch file
files = dir(fullfile(cfg.patchDir,'lesion_*.mat'));
n = numel(files);
fprintf('Found %d extracted lesion patch files.\n', n);
if n == 0
    error('No patch files found. Run s03_extract_patches.m first.');
end

fileNames = cell(n,1);
pids      = strings(n,1);
fids      = zeros(n,1);
y         = zeros(n,1);

for k = 1:n
    d = load(fullfile(files(k).folder, files(k).name), 'row');
    fileNames{k} = files(k).name;
    pids(k)      = strtrim(string(d.row.ProxID));
    fids(k)      = d.row.fid;
    v = d.row.ClinSig;
    if iscell(v), v = v{1}; end
    if islogical(v) || isnumeric(v)
        y(k) = double(v > 0);
    else
        y(k) = double(any(strcmpi(strtrim(char(v)), {'true','1'})));
    end
end

T_lesions = table(fileNames, cellstr(pids), fids, y, ...
    'VariableNames', {'FileName','ProxID','LesionID','ClinSig'});

fprintf('Total lesions: %d | Clinically significant: %d (%.1f%%) | Other: %d (%.1f%%)\n', ...
    n, sum(y==1), 100*mean(y==1), sum(y==0), 100*mean(y==0));

% ---- 2. Patient-level target: 1 if the patient has ANY significant lesion
up = unique(pids);
nP = numel(up);
patientTarget = zeros(nP,1);
for p = 1:nP
    patientTarget(p) = double(any(y(pids == up(p)) == 1));
end
fprintf('Patients: %d (%d with a significant lesion)\n\n', nP, sum(patientTarget));

% ---- 3. Stratified patient-level k-fold, repeated over seeds
splits = struct();
nLeak = 0;      % patients found in both train and validation
nBad  = 0;      % lesions not validated exactly once per seed

for si = 1:numel(cfg.seeds)
    seed = cfg.seeds(si);
    rng(seed);
    cv = cvpartition(patientTarget, 'KFold', cfg.nFolds);
    valCount = zeros(n,1);

    for f = 1:cfg.nFolds
        trainPats = up(cv.training(f));
        valPats   = up(cv.test(f));

        trainIdx = find(ismember(pids, trainPats));
        valIdx   = find(ismember(pids, valPats));

        nLeak = nLeak + numel(intersect(trainPats, valPats));
        valCount(valIdx) = valCount(valIdx) + 1;

        sp = struct('trainIdx', trainIdx, 'valIdx', valIdx, ...
                    'trainPats', {cellstr(trainPats)}, 'valPats', {cellstr(valPats)});
        splits.(sprintf('seed_%d',seed)).(sprintf('fold_%d',f)) = sp;
    end
    nBad = nBad + sum(valCount ~= 1);
end

% ---- 4. Save
save(fullfile(cfg.splitDir,'splits.mat'), 'splits', 'T_lesions');

% ---- 5. Report (leakage check is a REAL test)
sp1 = splits.seed_1;
fprintf('================= RESULT =================\n');
fprintf('Lesions in table            : %d (%d positive, %d negative)\n', n, sum(y==1), sum(y==0));
fprintf('Unique patients             : %d\n', nP);
fprintf('Splits                      : %d folds x %d seeds = %d\n', cfg.nFolds, numel(cfg.seeds), cfg.nFolds*numel(cfg.seeds));
fprintf('Seed 1 validation sizes     :');
for f = 1:cfg.nFolds
    vi = sp1.(sprintf('fold_%d',f)).valIdx;
    fprintf('  %d (%d pos)', numel(vi), sum(y(vi)==1));
end
fprintf('\n');
if nLeak == 0 && nBad == 0
    fprintf('Leakage check               : PASSED (0 shared patients, every lesion validated once per seed)\n');
else
    fprintf('Leakage check               : FAILED (shared patients: %d, bad lesion counts: %d)\n', nLeak, nBad);
end
fprintf('Saved to                    : %s\n', fullfile(cfg.splitDir,'splits.mat'));
fprintf('==========================================\n');