%% s15c_context_stats.m -- Experiment A: context size, patient-level bootstrap
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'T_lesions');
B  = load(fullfile(cfg.resDir,'baseline_oof.mat'));    % [n x 3 x 5]
C1 = load(fullfile(cfg.resDir,'cnn_oof.mat'));         % 24 mm
C2 = load(fullfile(cfg.resDir,'cnn_oof_ctx48.mat'));   % 48 mm
C3 = load(fullfile(cfg.resDir,'cnn_oof_ctx72.mat'));   % 72 mm
R  = load(fullfile(cfg.resDir,'resnet_oof.mat'));

y = double(T_lesions.ClinSig);  n = numel(y);
for f = {B,C1,C2,C3,R}
    assert(isequal(y(:), double(f{1}.y(:))), 'Labels differ between files');
end
nS = size(B.oof,3);
assert(size(C1.oof,2)==nS && size(C2.oof,2)==nS && size(C3.oof,2)==nS && size(R.oof,2)==nS, 'Seed counts differ');

names = {'ADC-only','Logistic (12 feat)','Random forest (12 feat)', ...
         'Small CNN 24 mm','Small CNN 48 mm','Small CNN 72 mm','ResNet-18 pretrained'};
nM = numel(names);
OOF = cat(2, B.oof, reshape(C1.oof,n,1,nS), reshape(C2.oof,n,1,nS), ...
             reshape(C3.oof,n,1,nS), reshape(double(R.oof),n,1,nS));

pt = zeros(nM,1);
for m = 1:nM
    a = zeros(1,nS);
    for s = 1:nS, a(s) = fastAUC(OOF(:,m,s), y); end
    pt(m) = mean(a);
end

pids = string(T_lesions.ProxID);
[up,~,g] = unique(pids);  nP = numel(up);
idxOf = accumarray(g, (1:n)', [], @(v){v});
nB = 2000;  rng(123);
bs = zeros(nB, nM);
fprintf('Bootstrapping %d patients x %d replicates...\n', nP, nB);
for b = 1:nB
    idx = vertcat(idxOf{randi(nP, nP, 1)});  yb = y(idx);
    for m = 1:nM
        acc = 0;
        for s = 1:nS, acc = acc + fastAUC(OOF(idx,m,s), yb); end
        bs(b,m) = acc / nS;
    end
end
ci = prctile(bs,[2.5 97.5]);  lo = ci(1,:)';  hi = ci(2,:)';

fprintf('\n================= RESULT =================\n');
fprintf('%d lesions (%d positive), %d patients, %d seeds x 5 folds\n\n', n, sum(y), nP, nS);
fprintf('%-26s  AUC     95%% CI (patient bootstrap)\n','Model');
for m = 1:nM
    fprintf('%-26s  %.3f   [%.3f, %.3f]\n', names{m}, pt(m), lo(m), hi(m));
end

% Pre-specified comparisons for experiment A (family of 2, Holm-corrected)
pairs = [4 5; 4 6];
nPair = size(pairs,1);
dd = zeros(nPair,1); dlo = dd; dhi = dd; pr = dd;
for k = 1:nPair
    d = bs(:,pairs(k,1)) - bs(:,pairs(k,2));
    dd(k) = pt(pairs(k,1)) - pt(pairs(k,2));
    q = prctile(d,[2.5 97.5]);  dlo(k) = q(1);  dhi(k) = q(2);
    pr(k) = max(min(1, 2*min(mean(d<=0), mean(d>=0))), 1/nB);
end
[psrt, ord] = sort(pr);  adj = zeros(nPair,1);  runMax = 0;
for j = 1:nPair
    runMax = max(runMax, min(1, (nPair-j+1)*psrt(j)));  adj(ord(j)) = runMax;
end
fprintf('\nExperiment A, paired differences (24 mm minus larger field of view):\n');
for k = 1:nPair
    fprintf('  %-16s minus %-16s  %+.3f  [%+.3f, %+.3f]   p %.3f   p(Holm) %.3f\n', ...
        names{pairs(k,1)}, names{pairs(k,2)}, dd(k), dlo(k), dhi(k), pr(k), adj(k));
end
fprintf('(CI/p reflect test-sample uncertainty only)\n');
fprintf('==========================================\n');

T = table(string(names(:)), pt, lo, hi, 'VariableNames', {'Model','AUC','CI_low','CI_high'});
writetable(T, fullfile(cfg.resDir,'auc_table_context.csv'));

figure('Color','w','Position',[100 100 720 650]); hold on;
sel = [1 4 5 6];  cols = lines(numel(sel));
for i = 1:numel(sel)
    m = sel(i);
    [fx,fy] = perfcurve(y, mean(OOF(:,m,:),3), 1);
    plot(fx, fy, 'LineWidth', 2, 'Color', cols(i,:), 'DisplayName', ...
         sprintf('%s: %.3f (%.3f-%.3f)', names{m}, pt(m), lo(m), hi(m)));
end
plot([0 1],[0 1],'k:','HandleVisibility','off');
xlabel('1 - Specificity'); ylabel('Sensitivity'); grid on; axis square;
legend('Location','southeast','FontSize',8);
title('Effect of field of view (small CNN, out-of-fold)');
print(gcf, fullfile(cfg.figDir,'roc_context.png'), '-dpng', '-r300');
fprintf('Saved: auc_table_context.csv and roc_context.png\n');

%% ---- helper
function auc = fastAUC(score, label)
pos = label == 1;  np = sum(pos);  nn = sum(~pos);
r = tiedrank(score(:));
auc = (sum(r(pos)) - np*(np+1)/2) / (np*nn);
end