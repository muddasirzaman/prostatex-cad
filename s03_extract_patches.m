%% ========================================================================
%  s03_extract_patches.m -- Geometry-correct 24 mm patches for T2 / ADC / DWI
%  RUN:    press the green Run button
%  TIME:   roughly 5-15 minutes
%  OUTPUT: study\patches\lesion_<ProxID>_L<fid>.mat  (one file per lesion)
%% ========================================================================
clear; clc;
cfg = s00_config();

load(fullfile(cfg.work,'series_index.mat'),'S');
S.ProxID   = string(S.ProxID);
S.Sequence = string(S.Sequence);

F = readtable(cfg.findingsCSV);
need = {'ProxID','fid','pos','ClinSig'};
if ~all(ismember(need, F.Properties.VariableNames))
    disp(F.Properties.VariableNames);
    error('Findings CSV must contain: ProxID, fid, pos, ClinSig');
end
pids = unique(string(F.ProxID),'stable');
fprintf('Findings: %d lesions from %d patients\n\n', height(F), numel(pids));

delete(fullfile(cfg.patchDir,'lesion_*.mat'));   % start clean, no stale files

nOK   = 0;
skips = strings(0,1);
qc    = zeros(0,8);   % [label t2C adcC dwiC dz_t2 dz_adc dz_dwi maxOob]
t0    = tic;

for p = 1:numel(pids)
    pid   = pids(p);
    cache = containers.Map('KeyType','char','ValueType','any');  % headers per series

    for r = find(string(F.ProxID)==pid)'
        fid   = F.fid(r);
        C     = parsePos(F.pos(r));
        label = parseLabel(F.ClinSig(r));
        if numel(C) ~= 3
            skips(end+1,1) = "unparsable pos"; continue; %#ok<SAGROW>
        end

        lesionPatches = struct();
        dz = nan(1,3); oob = zeros(1,3); reason = "";
        for s = 1:3
            seq     = cfg.sequences{s};
            folders = S.Folder(S.ProxID==pid & S.Sequence==seq);
            [patch, dz(s), oob(s), reason] = localPatch(folders, C, cfg, cache);
            if reason ~= ""
                reason = seq + ": " + reason;
                break;
            end
            lesionPatches.(seq) = patch;
        end
        if reason ~= ""
            skips(end+1,1) = reason; continue; %#ok<SAGROW>
        end

        row = F(r,:);            %#ok<NASGU>  (kept for s04)
        worldCoord = C;          %#ok<NASGU>
        qcInfo = struct('dz',dz,'oobFrac',oob); %#ok<NASGU>
        save(fullfile(cfg.patchDir, sprintf('lesion_%s_L%d_r%03d.mat', pid, fid, r)), ...
             'lesionPatches','row','worldCoord','qcInfo');

        cc = 25:40;   % central 16x16 px (about 6 mm)
        qc(end+1,:) = [label, ...
            mean2(lesionPatches.t2(cc,cc)), mean2(lesionPatches.adc(cc,cc)), ...
            mean2(lesionPatches.dwi(cc,cc)), dz, max(oob)]; %#ok<SAGROW>
        nOK = nOK + 1;
    end

    if mod(p,20)==0
        fprintf('  %3d/%d patients   %d lesions saved   %.1f min\n', ...
                p, numel(pids), nOK, toc(t0)/60);
    end
end

%% ------------------------------ REPORT ---------------------------------
fprintf('\n================= RESULT =================\n');
fprintf('Lesions in CSV      : %d\n', height(F));
fprintf('Lesions extracted   : %d\n', nOK);
fprintf('Lesions skipped     : %d\n', numel(skips));
if ~isempty(skips)
    [u,~,ic] = unique(skips); cnt = accumarray(ic,1);
    for k = 1:numel(u), fprintf('     %3d x  %s\n', cnt(k), u(k)); end
end
if nOK > 0
    fprintf('\nDistance from lesion to nearest slice (mm):\n');
    nm = {'t2','adc','dwi'};
    for s = 1:3
        fprintf('   %-4s median %.2f   max %.2f\n', nm{s}, ...
                median(qc(:,4+s)), max(qc(:,4+s)));
    end
    fprintf('Patches with >10%% outside image: %d\n', sum(qc(:,8) > 0.10));
    fprintf('Positives (ClinSig=1): %d   Negatives: %d\n', ...
            sum(qc(:,1)==1), sum(qc(:,1)==0));

    if numel(unique(qc(:,1))) == 2
        [~,~,~,aADC] = perfcurve(qc(:,1), -qc(:,3), 1);  % low ADC = cancer
        [~,~,~,aDWI] = perfcurve(qc(:,1),  qc(:,4), 1);  % high DWI = cancer
        [~,~,~,aT2 ] = perfcurve(qc(:,1), -qc(:,2), 1);  % dark T2 = cancer
        fprintf('\nSANITY: single-number AUC from the lesion centre\n');
        fprintf('   central ADC : %.3f   <-- most important\n', aADC);
        fprintf('   central DWI : %.3f\n', aDWI);
        fprintf('   central T2  : %.3f   (raw T2 is not calibrated; expect weak)\n', aT2);
    end
end
fprintf('Total time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');

%% ------------------------------ helpers --------------------------------
function C = parsePos(v)
s = char(string(v));
s = strrep(strrep(strrep(s,'[',' '),']',' '),',',' ');
C = sscanf(s,'%f')';
end

function y = parseLabel(v)
if iscell(v), v = v{1}; end
if islogical(v) || isnumeric(v)
    y = double(v > 0);
else
    y = double(any(strcmpi(strtrim(char(v)), {'true','1'})));
end
end

function g = localGeometry(folder)
% Read every header in one series folder once.
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
g.rowDir = iop(1:3);          % direction along a row (column index increases)
g.colDir = iop(4:6);          % direction down a column (row index increases)
g.normal = cross(g.rowDir, g.colDir);
g.proj   = g.ipp * g.normal';
g.rowSp  = ps(1);             % spacing between rows
g.colSp  = ps(2);             % spacing between columns
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
    i0 = dot(d, g.rowDir) / g.colSp;      % column position (0-based)
    j0 = dot(d, g.colDir) / g.rowSp;      % row position (0-based)
    inside = i0>=0 && i0<=g.nCol-1 && j0>=0 && j0<=g.nRow-1;
    score  = dz + 1000*(~inside);
    if score < bestScore
        bestScore = score;
        best = struct('g',g,'idx',idx,'dz',dz,'i0',i0,'j0',j0,'inside',inside);
    end
end

if isempty(best),        reason = "no readable series"; return; end
if ~best.inside,         reason = "lesion outside image"; return; end
if best.dz > 5,          reason = string(sprintf('nearest slice %.1f mm away', best.dz)); return; end

dzBest = best.dz;
fp   = best.g.files{best.idx};
info = dicominfo(fp);
img  = double(dicomread(fp));
if isfield(info,'RescaleSlope'),     img = img * double(info.RescaleSlope); end
if isfield(info,'RescaleIntercept'), img = img + double(info.RescaleIntercept); end

% Sample a true cfg.patchMM x cfg.patchMM window on a cfg.patchPx grid
step = cfg.patchMM / cfg.patchPx;
u = ((0:cfg.patchPx-1) - (cfg.patchPx-1)/2) * step;   % mm, centred
[U,V] = meshgrid(u,u);
Ig = best.i0 + U / best.g.colSp;
Jg = best.j0 + V / best.g.rowSp;
patch = interp2(img, Ig+1, Jg+1, 'linear', NaN);
bad = isnan(patch);
oobFrac = mean(bad(:));
if all(bad(:)), reason = "empty patch"; return; end
patch(bad) = median(patch(~bad));
end