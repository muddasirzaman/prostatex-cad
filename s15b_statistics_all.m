%% s15b_statistics_all.m -- All 5 models: patient-level bootstrap, paired tests, ROC
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'T_lesions');
B = load(fullfile(cfg.resDir,'baseline_oof.mat'));   % oof [n x 3 x 5]
C = load(fullfile(cfg.resDir,'cnn_oof.mat'));        % oof [n x 5]
R = load(fullfile(cfg.resDir,'resnet_oof.mat'));     % oof [n x 5]  (from Python)

y = double(T_lesions.ClinSig);
n = numel(y);
assert(isequal(y(:), double(B.y(:))) && isequal(y(:), double(C.y(:))) && isequal(y(:), double(R.y(:))), ...
       'Labels differ between files');
nS = size(B.oof,3);
assert(size(C.oof,2)==nS && size(R.oof,2)==nS, 'Different numbers of seeds across models');

names = {'ADC-only','Logistic (12 feat)','Random forest (12 feat)','Small CNN','ResNet-18 pretrained'};
nM = numel(names);
OOF = cat(2, B.oof, reshape(C.oof,n,1,nS), reshape(double(R.oof),n,1,nS));   % n x 5 x nSeeds

% ---- Point estimates (mean over seeds of per-seed pooled AUC)
pt = zeros(nM,1);
for m = 1:nM
    a = zeros(1,nS);
    for s = 1:nS, a(s) = fastAUC(OOF(:,m,s), y); end
    pt(m) = mean(a);
end

% ---- Patient-level bootstrap
pids = string(T_lesions.ProxID);
[up,~,g] = unique(pids);
nP = numel(up);
idxOf = accumarray(g, (1:n)', [], @(v){v});
nB = 2000;
rng(123);
bs = zeros(nB, nM);
fprintf('Bootstrapping %d patients x %d replicates...\n', nP, nB);
for b = 1:nB
    ps  = randi(nP, nP, 1);
    idx = vertcat(idxOf{ps});
    yb  = y(idx);
    for m = 1:nM
        acc = 0;
        for s = 1:nS, acc = acc + fastAUC(OOF(idx,m,s), yb); end
        bs(b,m) = acc / nS;
    end
end
ci = prctile(bs,[2.5 97.5]);  lo = ci(1,:)';  hi = ci(2,:)';

% ---- Report
fprintf('\n================= RESULT =================\n');
fprintf('%d lesions (%d positive) from %d patients; %d seeds x 5 folds\n\n', n, sum(y), nP, nS);
fprintf('%-26s  AUC     95%% CI (patient bootstrap)\n','Model');
for m = 1:nM
    fprintf('%-26s  %.3f   [%.3f, %.3f]\n', names{m}, pt(m), lo(m), hi(m));
end

pairs = [4 1; 4 2; 4 3; 4 5; 3 1; 5 1];
nPair = size(pairs,1);
dd = zeros(nPair,1); dlo = dd; dhi = dd; pr = dd;
for k = 1:nPair
    d = bs(:,pairs(k,1)) - bs(:,pairs(k,2));
    dd(k) = pt(pairs(k,1)) - pt(pairs(k,2));
    q = prctile(d,[2.5 97.5]); dlo(k) = q(1); dhi(k) = q(2);
    pr(k) = max(min(1, 2*min(mean(d<=0), mean(d>=0))), 1/nB);
end
% Holm correction across the comparisons
[ps_sorted, ord] = sort(pr);
adj = zeros(nPair,1);
runMax = 0;
for j = 1:nPair
    runMax = max(runMax, min(1, (nPair-j+1)*ps_sorted(j)));
    adj(ord(j)) = runMax;
end

fprintf('\nPaired differences (same bootstrap samples):\n');
fprintf('  %-40s  diff    95%% CI              p     p(Holm)\n','Comparison');
for k = 1:nPair
    fprintf('  %-18s minus %-18s  %+.3f  [%+.3f, %+.3f]   %.3f  %.3f\n', ...
        names{pairs(k,1)}, names{pairs(k,2)}, dd(k), dlo(k), dhi(k), pr(k), adj(k));
end
fprintf('(CI/p reflect test-sample uncertainty only; Holm adjusts for %d comparisons)\n', nPair);
fprintf('==========================================\n');

% ---- Save table + ROC figure
T = table(string(names(:)), pt, lo, hi, 'VariableNames', {'Model','AUC','CI_low','CI_high'});
writetable(T, fullfile(cfg.resDir,'auc_table_all.csv'));

figure('Color','w','Position',[100 100 720 650]); hold on;
cols = lines(nM);
for m = 1:nM
    [fx,fy] = perfcurve(y, mean(OOF(:,m,:),3), 1);
    plot(fx, fy, 'LineWidth', 2, 'Color', cols(m,:), 'DisplayName', ...
         sprintf('%s: %.3f (%.3f-%.3f)', names{m}, pt(m), lo(m), hi(m)));
end
plot([0 1],[0 1],'k:','HandleVisibility','off');
xlabel('1 - Specificity'); ylabel('Sensitivity'); grid on; axis square;
legend('Location','southeast','FontSize',8);
title('Out-of-fold ROC, ProstateX training set (330 lesions)');
print(gcf, fullfile(cfg.figDir,'roc_comparison_all.png'), '-dpng', '-r300');
fprintf('Saved: auc_table_all.csv and roc_comparison_all.png\n');

%% ---- helper
function auc = fastAUC(score, label)
pos = label == 1;  np = sum(pos);  nn = sum(~pos);
r = tiedrank(score(:));
auc = (sum(r(pos)) - np*(np+1)/2) / (np*nn);
end