%% s19_evaluate_test.m -- ONE-TIME evaluation on the independent ProstateX test set
%  All models were trained in s18 on the 330 training lesions with settings fixed in advance.
clear; clc;
cfg = s00_config();
modelFile = fullfile(cfg.modelDir,'final_models.mat');
assert(exist(modelFile,'file')==2, 'Run s18_train_final_models.m first');
M = load(modelFile);                       % m2 m3 mu sd nets adcMu adcSd

files = dir(fullfile(cfg.work,'patches_test','test_*.mat'));
n = numel(files);
fprintf('Test patch files: %d\n', n);
Xf = zeros(n,12);  X = zeros(cfg.patchPx, cfg.patchPx, 3, n, 'single');
y = zeros(n,1);    pid = strings(n,1);
for k = 1:n
    d = load(fullfile(files(k).folder, files(k).name));
    Xf(k,:) = makeFeatures(d.lesionPatches);
    t2  = single(d.lesionPatches.t2);
    adc = single(d.lesionPatches.adc);
    dwi = single(d.lesionPatches.dwi);
    X(:,:,1,k) = (t2  - mean(t2(:)))  / (std(t2(:))  + 1e-6);
    X(:,:,2,k) = (adc - M.adcMu) / M.adcSd;          % training-set ADC statistics
    X(:,:,3,k) = (dwi - mean(dwi(:))) / (std(dwi(:)) + 1e-6);
    y(k)   = d.label;
    pid(k) = string(d.row.ProxID);
end

% ---- Scores
names = {'ADC-only (rule)','Logistic (12 feat)','Random forest (12 feat)','Small CNN (5-net average)'};
Xs = (Xf - M.mu) ./ M.sd;
sc = zeros(n,4);
sc(:,1) = -Xf(:,5);                                   % lower central ADC = more likely cancer
[~,p] = predict(M.m2, Xs);  sc(:,2) = p(:,2);
[~,p] = predict(M.m3, Xs);  sc(:,3) = p(:,2);
pc = zeros(n,1);
for s = 1:numel(M.nets)
    p = predict(M.nets{s}, X);  pc = pc + p(:,2) / numel(M.nets);
end
sc(:,4) = pc;

% ---- Diffusion protocol group (decided from image headers, not labels)
load(fullfile(cfg.work,'series_index_test.mat'),'S');
S.ProxID = string(S.ProxID);  S.Sequence = string(S.Sequence);
grp = strings(n,1);
for k = 1:n
    r = find(S.ProxID==pid(k) & S.Sequence=="adc", 1);
    dsc = string(S.SeriesDesc(r));
    if contains(dsc,'DYNDIST') || contains(dsc,'4bval_fs_ADC')
        grp(k) = "A: protocols also in training";
    else
        grp(k) = "B: protocols not in training";
    end
end

% ---- Overall AUC with patient-level bootstrap
nB = 2000;
[pt, lo, hi, bs] = bootAUC(sc, y, pid, nB, 123);
fprintf('\n================= RESULT =================\n');
fprintf('Independent test set: %d lesions (%d positive), %d patients\n\n', n, sum(y), numel(unique(pid)));
fprintf('%-26s  AUC     95%% CI (patient bootstrap)\n','Model');
for m = 1:4, fprintf('%-26s  %.3f   [%.3f, %.3f]\n', names{m}, pt(m), lo(m), hi(m)); end

% ---- Pre-specified paired comparisons: CNN vs each other model, Holm over 3
pairs = [4 1; 4 2; 4 3];
dd = zeros(3,1); dlo = dd; dhi = dd; pr = dd;
for k = 1:3
    d = bs(:,pairs(k,1)) - bs(:,pairs(k,2));  d = d(~isnan(d));
    dd(k) = pt(pairs(k,1)) - pt(pairs(k,2));
    q = prctile(d,[2.5 97.5]);  dlo(k) = q(1);  dhi(k) = q(2);
    pr(k) = max(min(1, 2*min(mean(d<=0), mean(d>=0))), 1/numel(d));
end
[ps_sorted, ord] = sort(pr);  adj = zeros(3,1);  runMax = 0;
for j = 1:3
    runMax = max(runMax, min(1, (3-j+1)*ps_sorted(j)));  adj(ord(j)) = runMax;
end
fprintf('\nPaired differences (CNN minus other):\n');
for k = 1:3
    fprintf('  CNN minus %-24s %+.3f  [%+.3f, %+.3f]   p %.3f   p(Holm) %.3f\n', ...
        names{pairs(k,2)}, dd(k), dlo(k), dhi(k), pr(k), adj(k));
end

% ---- By diffusion protocol
fprintf('\nBy diffusion protocol (CIs are wide: small groups):\n');
for g = unique(grp)'
    idx = grp == g;
    if numel(unique(y(idx))) < 2, continue; end
    [pg, lg, hg] = bootAUC(sc(idx,:), y(idx), pid(idx), nB, 321);
    fprintf('  %s: %d lesions (%d positive)\n', g, sum(idx), sum(y(idx)));
    for m = 1:4, fprintf('     %-26s %.3f  [%.3f, %.3f]\n', names{m}, pg(m), lg(m), hg(m)); end
end
fprintf('(CIs reflect test-sample uncertainty only)\n');
fprintf('==========================================\n');

% ---- Save results and ROC figure
T = table(string(names(:)), pt, lo, hi, 'VariableNames', {'Model','AUC','CI_low','CI_high'});
writetable(T, fullfile(cfg.resDir,'auc_table_test.csv'));
save(fullfile(cfg.resDir,'test_scores.mat'), 'sc','y','pid','grp','names');
figure('Color','w','Position',[100 100 720 650]); hold on;
cols = lines(4);
for m = 1:4
    [fx,fy] = perfcurve(y, sc(:,m), 1);
    plot(fx, fy, 'LineWidth', 2, 'Color', cols(m,:), 'DisplayName', ...
         sprintf('%s: %.3f (%.3f-%.3f)', names{m}, pt(m), lo(m), hi(m)));
end
plot([0 1],[0 1],'k:','HandleVisibility','off');
xlabel('1 - Specificity'); ylabel('Sensitivity'); grid on; axis square;
legend('Location','southeast','FontSize',8);
title('Independent test set ROC (ProstateX test lesions)');
print(gcf, fullfile(cfg.figDir,'roc_test.png'), '-dpng', '-r300');
fprintf('Saved: auc_table_test.csv, test_scores.mat, roc_test.png\n');

%% ---- helpers
function f = makeFeatures(lp)
seq = {'t2','adc','dwi'};  f = zeros(1,12);
for s = 1:3
    im  = double(lp.(seq{s}));
    c1  = im(25:40,25:40);  c2 = im(17:48,17:48);  med = median(im(:));
    f((s-1)*4 + (1:4)) = [mean(c1(:)), mean(c2(:)), std(c2(:)), mean(c1(:)) / (abs(med) + 1e-6)];
end
end

function [pt, lo, hi, bs] = bootAUC(sc, y, pid, nB, seed)
[up,~,g] = unique(pid);  nP = numel(up);
idxOf = accumarray(g, (1:numel(y))', [], @(v){v});
m = size(sc,2);
pt = zeros(m,1);
for j = 1:m, pt(j) = fastAUC(sc(:,j), y); end
rng(seed);  bs = nan(nB, m);
for b = 1:nB
    idx = vertcat(idxOf{randi(nP, nP, 1)});  yb = y(idx);
    if all(yb==0) || all(yb==1), continue; end
    for j = 1:m, bs(b,j) = fastAUC(sc(idx,j), yb); end
end
ci = prctile(bs,[2.5 97.5]);  lo = ci(1,:)';  hi = ci(2,:)';
end

function auc = fastAUC(score, label)
pos = label == 1;  np = sum(pos);  nn = sum(~pos);
r = tiedrank(score(:));
auc = (sum(r(pos)) - np*(np+1)/2) / (np*nn);
end