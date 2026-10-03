%% s17_extract_test_patches.m -- Independent test set: index series + extract 24 mm patches
%  Uses test-specific sequence patterns. Training config and index are NOT modified.
clear; clc;
cfg = s00_config();

pat.adc = {'DYNDIST_MIX_ADC','DYNDIST_ADC','4bval_fs_ADC','alle spoelen_ADC','4bval_spair_511b_ADC'};
pat.dwi = {'DYNDIST_MIXCALC_BVAL','DYNDISTCALC_BVAL','4bval_fsCALC_BVAL','alle spoelenCALC_BVAL','4bval_spair_511bCALC_BVAL'};
pat.t2  = {'t2_tse_tra'};
exclude = cfg.patExclude;

testFile = 'D:\prostate cance\test_info\Findings-Test-merged.csv';
outDir   = fullfile(cfg.work,'patches_test');
if ~exist(outDir,'dir'), mkdir(outDir); end
delete(fullfile(outDir,'test_*.mat'));

T = readtable(testFile);
pids = unique(string(T.ProxID),'stable');
fprintf('Test findings: %d lesions, %d patients\n\n', height(T), numel(pids));

%% ---- 1. Build the series index for the test patients
rows = {};  t0 = tic;
for p = 1:numel(pids)
    dcm = dir(fullfile(cfg.dicomRoot, char(pids(p)), '**', '*.dcm'));
    if isempty(dcm), continue; end
    folders = unique({dcm.folder});
    for s = 1:numel(folders)
        f = dir(fullfile(folders{s},'*.dcm'));
        if isempty(f), continue; end
        try, info = dicominfo(fullfile(f(1).folder,f(1).name)); catch, continue; end
        desc = '';  if isfield(info,'SeriesDescription'), desc = info.SeriesDescription; end
        key = classifySeries(desc, pat, exclude);
        if isempty(key), continue; end
        rows(end+1,:) = {char(pids(p)), key, string(desc), numel(f), folders{s}}; %#ok<SAGROW>
    end
    if mod(p,20)==0, fprintf('  indexed %d/%d patients   %.1f min\n', p, numel(pids), toc(t0)/60); end
end
S = cell2table(rows,'VariableNames',{'ProxID','Sequence','SeriesDesc','nFiles','Folder'});
S.ProxID = string(S.ProxID);  S.Sequence = string(S.Sequence);
save(fullfile(cfg.work,'series_index_test.mat'),'S');

fprintf('\nSeries names matched, per sequence:\n');
for q = {'adc','dwi'}
    u = unique(S.SeriesDesc(S.Sequence==q{1}));
    fprintf('  %s: %s\n', upper(q{1}), strjoin(cellstr(u'), ' | '));
end

%% ---- 2. Extract 24 mm patches
cfgT = cfg;                                  % patchMM = 24, patchPx = 64 as in training
nOK = 0;  skips = strings(0,1);  skipLab = zeros(0,1);
okLab = zeros(0,1);  okGrp = strings(0,1);  qc = zeros(0,5);  dzAll = zeros(0,3);
for p = 1:numel(pids)
    pid   = pids(p);
    cache = containers.Map('KeyType','char','ValueType','any');
    for r = find(string(T.ProxID)==pid)'
        fid   = T.fid(r);
        C     = parsePos(T.pos(r));
        label = double(strcmpi(strtrim(char(string(T.ClinSig(r)))),'true'));
        if numel(C) ~= 3
            skips(end+1,1) = "unparsable pos"; skipLab(end+1,1) = label; continue; %#ok<SAGROW>
        end
        lesionPatches = struct();  dz = nan(1,3);  oob = zeros(1,3);  reason = "";
        for s = 1:3
            seq     = cfg.sequences{s};
            folders = S.Folder(S.ProxID==pid & S.Sequence==seq);
            [patch, dz(s), oob(s), reason] = localPatch(folders, C, cfgT, cache);
            if reason ~= "", reason = seq + ": " + reason; break; end
            lesionPatches.(seq) = patch;
        end
        if reason ~= ""
            skips(end+1,1) = reason; skipLab(end+1,1) = label; continue; %#ok<SAGROW>
        end
        row = T(r,:);  worldCoord = C;  qcInfo = struct('dz',dz,'oobFrac',oob); %#ok<NASGU>
        save(fullfile(outDir, sprintf('test_%s_L%d_r%03d.mat', pid, fid, r)), ...
             'lesionPatches','row','worldCoord','qcInfo','label');
        nOK = nOK + 1;
        okLab(end+1,1) = label; okGrp(end+1,1) = pid; dzAll(end+1,:) = dz; %#ok<SAGROW>
        cc = 25:40;
        qc(end+1,:) = [mean2(lesionPatches.t2(cc,cc)), mean2(lesionPatches.adc(cc,cc)), ...
                       mean2(lesionPatches.dwi(cc,cc)), max(oob), 0]; %#ok<SAGROW>
    end
end

%% ---- 3. Report
fprintf('\n================= RESULT =================\n');
fprintf('Test lesions in CSV      : %d (%d positive)\n', height(T), sum(strcmpi(string(T.ClinSig),"true")));
fprintf('Test lesions extracted   : %d (%d positive, %d negative)\n', nOK, sum(okLab==1), sum(okLab==0));
fprintf('Test lesions excluded    : %d (%d positive, %d negative)\n', numel(skips), sum(skipLab==1), sum(skipLab==0));
if ~isempty(skips)
    [u,~,ic] = unique(skips);  cnt = accumarray(ic,1);
    for k = 1:numel(u), fprintf('     %3d x  %s\n', cnt(k), u(k)); end
end
if nOK > 0
    fprintf('Max slice distance (mm)  : %.2f\n', max(dzAll(:)));
    fprintf('Patches with >10%% outside image: %d\n', sum(qc(:,4) > 0.10));
    if numel(unique(okLab)) == 2
        [~,~,~,aA] = perfcurve(okLab, -qc(:,2), 1);
        [~,~,~,aD] = perfcurve(okLab,  qc(:,3), 1);
        fprintf('\nQC (fixed rule, no training): central ADC AUC %.3f | central DWI AUC %.3f\n', aA, aD);
        fprintf('   (training set gave 0.740 / 0.718)\n');
    end
end
fprintf('Total time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');

%% ------------------------------ helpers --------------------------------
function key = classifySeries(desc, pat, exclude)
key = '';
d = lower(string(desc));
for e = exclude
    if contains(d, lower(e{1})), return; end
end
for f = {'adc','dwi','t2'}
    for q = 1:numel(pat.(f{1}))
        if contains(d, lower(pat.(f{1}){q})), key = f{1}; return; end
    end
end
end

function C = parsePos(v)
s = char(string(v));
s = strrep(strrep(strrep(s,'[',' '),']',' '),',',' ');
C = sscanf(s,'%f')';
end

function g = localGeometry(folder)
g = [];
f = dir(fullfile(folder,'*.dcm'));
n = numel(f);
if n == 0, return; end
ipp = nan(n,3); ok = false(n,1); files = cell(n,1);
for k = 1:n
    try
        fp   = fullfile(f(k).folder, f(k).name);
        info = dicominfo(fp);
        ipp(k,:) = info.ImagePositionPatient(:)';
        if ~any(ok)
            iop  = info.ImageOrientationPatient(:)';
            ps   = info.PixelSpacing(:)';
            nRow = double(info.Rows);  nCol = double(info.Columns);
        end
        files{k} = fp; ok(k) = true;
    catch
    end
end
if ~any(ok), return; end
g.ipp    = ipp(ok,:);
g.files  = files(ok);
g.rowDir = iop(1:3);
g.colDir = iop(4:6);
g.normal = cross(g.rowDir, g.colDir);
g.proj   = g.ipp * g.normal';
g.rowSp  = ps(1);
g.colSp  = ps(2);
g.nRow   = nRow;  g.nCol = nCol;
end

function [patch, dzBest, oobFrac, reason] = localPatch(folders, C, cfg, cache)
patch = []; dzBest = NaN; oobFrac = 0; reason = "";
if isempty(folders), reason = "no series in index"; return; end
best = []; bestScore = inf;
for q = 1:numel(folders)
    key = folders{q};
    if ~isKey(cache,key), cache(key) = localGeometry(key); end %#ok<NASGU>
    g = cache(key);
    if isempty(g), continue; end
    [dz, idx] = min(abs(g.proj - dot(C, g.normal)));
    d  = C - g.ipp(idx,:);
    i0 = dot(d, g.rowDir) / g.colSp;
    j0 = dot(d, g.colDir) / g.rowSp;
    inside = i0>=0 && i0<=g.nCol-1 && j0>=0 && j0<=g.nRow-1;
    score  = dz + 1000*(~inside);
    if score < bestScore
        bestScore = score;
        best = struct('g',g,'idx',idx,'dz',dz,'i0',i0,'j0',j0,'inside',inside);
    end
end
if isempty(best),  reason = "no readable series"; return; end
if ~best.inside,   reason = "lesion outside image"; return; end
if best.dz > 5,    reason = string(sprintf('nearest slice %.1f mm away', best.dz)); return; end
dzBest = best.dz;
fp   = best.g.files{best.idx};
info = dicominfo(fp);
img  = double(dicomread(fp));
if isfield(info,'RescaleSlope'),     img = img * double(info.RescaleSlope); end
if isfield(info,'RescaleIntercept'), img = img + double(info.RescaleIntercept); end
step = cfg.patchMM / cfg.patchPx;
u = ((0:cfg.patchPx-1) - (cfg.patchPx-1)/2) * step;
[U,V] = meshgrid(u,u);
Ig = best.i0 + U / best.g.colSp;
Jg = best.j0 + V / best.g.rowSp;
patch = interp2(img, Ig+1, Jg+1, 'linear', NaN);
bad = isnan(patch);
oobFrac = mean(bad(:));
if all(bad(:)), reason = "empty patch"; return; end
patch(bad) = median(patch(~bad));
end