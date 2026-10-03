%% s03b_extract_context_patches.m -- Same lesions, larger field of view
clear; clc;
cfg = s00_config();

ctxMM = 72;                       % <-- run with 48 first, later set to 72 and run again
cfg.patchMM  = ctxMM;
cfg.patchDir = fullfile(cfg.work, sprintf('patches_%d', ctxMM));
if ~exist(cfg.patchDir,'dir'), mkdir(cfg.patchDir); end
delete(fullfile(cfg.patchDir,'lesion_*.mat'));

load(fullfile(cfg.work,'series_index.mat'),'S');
S.ProxID = string(S.ProxID);  S.Sequence = string(S.Sequence);
F = readtable(cfg.findingsCSV);
pids = unique(string(F.ProxID),'stable');
fprintf('Field of view %d mm -> %s\n\n', ctxMM, cfg.patchDir);

nOK = 0;  skips = strings(0,1);  oobAll = zeros(0,1);  dzAll = zeros(0,3);
t0 = tic;
for p = 1:numel(pids)
    pid   = pids(p);
    cache = containers.Map('KeyType','char','ValueType','any');
    for r = find(string(F.ProxID)==pid)'
        fid = F.fid(r);
        C   = parsePos(F.pos(r));
        if numel(C) ~= 3
            skips(end+1,1) = "unparsable pos"; continue; %#ok<SAGROW>
        end
        lesionPatches = struct();
        dz = nan(1,3);  oob = zeros(1,3);  reason = "";
        for s = 1:3
            seq     = cfg.sequences{s};
            folders = S.Folder(S.ProxID==pid & S.Sequence==seq);
            [patch, dz(s), oob(s), reason] = localPatch(folders, C, cfg, cache);
            if reason ~= ""
                reason = seq + ": " + reason; break;
            end
            lesionPatches.(seq) = patch;
        end
        if reason ~= ""
            skips(end+1,1) = reason; continue; %#ok<SAGROW>
        end
        row = F(r,:);  worldCoord = C;  qcInfo = struct('dz',dz,'oobFrac',oob);
        save(fullfile(cfg.patchDir, sprintf('lesion_%s_L%d_r%03d.mat', pid, fid, r)), ...
             'lesionPatches','row','worldCoord','qcInfo');
        nOK = nOK + 1;
        oobAll(end+1,1) = max(oob);   dzAll(end+1,:) = dz; %#ok<SAGROW>
    end
    if mod(p,20)==0
        fprintf('  %3d/%d patients   %d lesions saved   %.1f min\n', p, numel(pids), nOK, toc(t0)/60);
    end
end

fprintf('\n================= RESULT =================\n');
fprintf('Field of view            : %d mm (64x64 grid, %.3f mm/pixel)\n', ctxMM, ctxMM/cfg.patchPx);
fprintf('Lesions in CSV           : %d\n', height(F));
fprintf('Lesions extracted        : %d\n', nOK);
fprintf('Lesions skipped          : %d\n', numel(skips));
if nOK > 0
    fprintf('Max slice distance (mm)  : %.2f\n', max(dzAll(:)));
    fprintf('Patches with >10%% outside image: %d (max %.0f%% outside)\n', sum(oobAll>0.10), 100*max(oobAll));
end
fprintf('Total time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');

%% ------------------------------ helpers --------------------------------
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