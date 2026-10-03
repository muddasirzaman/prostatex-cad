%% s20_test_metrics.m -- Operating-point metrics, confusion matrices, PR curves on the test set
%  Thresholds are fixed on TRAINING out-of-fold scores, never on test labels.
clear; clc;
cfg = s00_config();
T  = load(fullfile(cfg.resDir,'test_scores.mat'));     % sc [n x 4], y, pid, grp, names
B  = load(fullfile(cfg.resDir,'baseline_oof.mat'));    % oof [330 x 3 x 5]
Cn = load(fullfile(cfg.resDir,'cnn_oof.mat'));         % oof [330 x 5]
sc = T.sc;  y = T.y(:);  pid = string(T.pid);  names = T.names;
ytr = double(B.y(:));
trS = [mean(B.oof(:,1,:),3), mean(B.oof(:,2,:),3), mean(B.oof(:,3,:),3), mean(Cn.oof,2)];
nM = 4;

% ---- Thresholds from training data: col 1 = 90% sensitivity, col 2 = Youden
thr = zeros(nM,2);
for m = 1:nM
    thr(m,1) = prctile(trS(ytr==1,m), 10);
    [fx,fy,Tt] = perfcurve(ytr, trS(:,m), 1);
    [~,i] = max(fy - fx);  thr(m,2) = Tt(i);
end
opNames = {'A: rule-out (90% sensitivity on training)','B: Youden (training)'};

% ---- Patient-level bootstrap for the metrics and average precision
nB = 2000;  rng(2024);
[up,~,g] = unique(pid);  nP = numel(up);
idxOf = accumarray(g, (1:numel(y))', [], @(v){v});
bsMet = nan(nB, 4, nM, 2);
bsAP  = nan(nB, nM);
for b = 1:nB
    idx = vertcat(idxOf{randi(nP, nP, 1)});
    for m = 1:nM
        bsAP(b,m) = avgPrec(sc(idx,m), y(idx));
        for o = 1:2
            r = metricsAt(sc(idx,m), y(idx), thr(m,o));
            bsMet(b,:,m,o) = r(5:8);
        end
    end
end
ci   = prctile(bsMet, [2.5 97.5], 1);      % 2 x 4 x nM x 2
apCI = prctile(bsAP,  [2.5 97.5], 1);      % 2 x nM

% ---- Report
fprintf('\n================= RESULT =================\n');
fprintf('Test set: %d lesions (%d positive; prevalence %.1f%%). Thresholds fixed on training data.\n', numel(y), sum(y), 100*mean(y));
for o = 1:2
    fprintf('\nOperating point %s\n', opNames{o});
    fprintf('%-26s  TP  FP  FN  TN   Sens [95%% CI]       Spec [95%% CI]       PPV [95%% CI]        NPV [95%% CI]\n','Model');
    for m = 1:nM
        r = metricsAt(sc(:,m), y, thr(m,o));
        c = squeeze(ci(:,:,m,o));
        fprintf('%-26s %3d %3d %3d %3d   %.2f [%.2f,%.2f]   %.2f [%.2f,%.2f]   %.2f [%.2f,%.2f]   %.2f [%.2f,%.2f]\n', ...
            names{m}, r(1),r(2),r(3),r(4), r(5),c(1,1),c(2,1), r(6),c(1,2),c(2,2), r(7),c(1,3),c(2,3), r(8),c(1,4),c(2,4));
    end
end
fprintf('\nAverage precision (area under the precision-recall curve; chance = prevalence %.2f)\n', mean(y));
for m = 1:nM
    fprintf('  %-26s %.3f  [%.3f, %.3f]\n', names{m}, avgPrec(sc(:,m), y), apCI(1,m), apCI(2,m));
end
fprintf('(CIs reflect test-sample uncertainty only; thresholds come from training out-of-fold scores)\n');
fprintf('==========================================\n');

% ---- Figures
cols = lines(nM);  mk = {'o','s'};
figure('Color','w','Position',[100 100 720 650]); hold on;
for m = 1:nM
    [fx,fy] = perfcurve(y, sc(:,m), 1);
    plot(fx, fy, 'LineWidth', 1.8, 'Color', cols(m,:), 'DisplayName', names{m});
    for o = 1:2
        r = metricsAt(sc(:,m), y, thr(m,o));
        plot(1-r(6), r(5), mk{o}, 'MarkerSize', 9, 'MarkerFaceColor', cols(m,:), 'MarkerEdgeColor','k', 'HandleVisibility','off');
    end
end
plot([0 1],[0 1],'k:','HandleVisibility','off');
xlabel('1 - Specificity'); ylabel('Sensitivity'); grid on; axis square;
legend('Location','southeast','FontSize',8);
title('Test ROC; circles = threshold A (90% sens.), squares = threshold B (Youden)');
print(gcf, fullfile(cfg.figDir,'roc_test_operating_points.png'), '-dpng', '-r300');

figure('Color','w','Position',[100 100 720 650]); hold on;
for m = 1:nM
    [rc,pr] = prCurve(sc(:,m), y);
    plot(rc, pr, 'LineWidth', 1.8, 'Color', cols(m,:), 'DisplayName', names{m});
end
yline(mean(y),'k:','Chance (prevalence)','HandleVisibility','off');
xlabel('Recall (sensitivity)'); ylabel('Precision (PPV)'); grid on; ylim([0 1]); axis square;
legend('Location','northeast','FontSize',8);
title('Test precision-recall curves');
print(gcf, fullfile(cfg.figDir,'pr_test.png'), '-dpng', '-r300');

figure('Color','w','Position',[100 100 900 420]);
cmap = [linspace(1,0.2,64)' linspace(1,0.45,64)' linspace(1,0.85,64)'];
for o = 1:2
    subplot(1,2,o);
    r = metricsAt(sc(:,4), y, thr(4,o));
    cmv = [r(4) r(2); r(3) r(1)];                 % rows: true NS / true S; cols: predicted NS / S
    imagesc(cmv); colormap(cmap); axis equal tight;
    set(gca,'XTick',[1 2],'XTickLabel',{'Pred. not sig.','Pred. sig.'},'YTick',[1 2],'YTickLabel',{'True not sig.','True sig.'});
    for i = 1:2, for j = 1:2
        text(j, i, num2str(cmv(i,j)), 'HorizontalAlignment','center', 'FontSize',18, 'Color','k');
    end, end
    title(sprintf('CNN, threshold %s', opNames{o}), 'FontSize', 9);
end
print(gcf, fullfile(cfg.figDir,'confusion_test_cnn.png'), '-dpng', '-r300');
save(fullfile(cfg.resDir,'test_metrics.mat'), 'thr','opNames','names','ci','apCI');
fprintf('Saved figures: roc_test_operating_points.png, pr_test.png, confusion_test_cnn.png\n');

%% ---- helpers
function r = metricsAt(score, label, thr)
pred = score >= thr;
tp = sum(pred & label==1);  fp = sum(pred & label==0);
fn = sum(~pred & label==1); tn = sum(~pred & label==0);
r = [tp fp fn tn tp/(tp+fn) tn/(tn+fp) tp/(tp+fp) tn/(tn+fn)];
end
function ap = avgPrec(score, label)
[~, ord] = sort(score(:), 'descend');
l = label(ord) == 1;  tp = cumsum(l);  k = (1:numel(l))';
ap = sum(tp(l) ./ k(l)) / sum(l);
end
function [rc, pr] = prCurve(score, label)
[~, ord] = sort(score(:), 'descend');
l = label(ord) == 1;  tp = cumsum(l);  k = (1:numel(l))';
rc = tp / sum(l);  pr = tp ./ k;
end