%% s13_baseline_all_splits.m -- Honest baseline over all 5 folds x 5 seeds
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'splits','T_lesions');
n = height(T_lesions);
y = double(T_lesions.ClinSig);
seq = {'t2','adc','dwi'};

% ---- 12 features: per sequence [mean 6mm, mean 12mm, std 12mm, 6mm/patch-median]
Xf = zeros(n,12);
for k = 1:n
    d = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    f = zeros(1,12);
    for s = 1:3
        im  = double(d.lesionPatches.(seq{s}));
        c1  = im(25:40,25:40);          % central 16x16 px (~6 mm)
        c2  = im(17:48,17:48);          % central 32x32 px (~12 mm)
        med = median(im(:));
        f((s-1)*4 + (1:4)) = [mean(c1(:)), mean(c2(:)), std(c2(:)), ...
                              mean(c1(:)) / (abs(med) + 1e-6)];
    end
    Xf(k,:) = f;
end

modelNames = {'ADC-only (no training)','Logistic (12 feat)','Random forest (12 feat)'};
nM = numel(modelNames);
seedNames = fieldnames(splits);
nS = numel(seedNames);
oof = nan(n, nM, nS);            % out-of-fold scores

rng(0);
for si = 1:nS
    foldNames = fieldnames(splits.(seedNames{si}));
    for fi = 1:numel(foldNames)
        sp  = splits.(seedNames{si}).(foldNames{fi});
        tr  = sp.trainIdx(:);  va = sp.valIdx(:);

        % Model 1: lower central ADC (6 mm mean) = more likely cancer
        oof(va,1,si) = -Xf(va,5);

        % Standardise with TRAIN statistics only
        mu = mean(Xf(tr,:)); sd = std(Xf(tr,:)) + 1e-9;
        Xtr = (Xf(tr,:) - mu) ./ sd;
        Xva = (Xf(va,:) - mu) ./ sd;

        % Model 2: ridge logistic regression, uniform class prior
        m2 = fitclinear(Xtr, y(tr), 'Learner','logistic', 'Regularization','ridge', ...
                        'Lambda',0.05, 'Prior','uniform', 'ClassNames',[0 1]);
        [~,sc] = predict(m2, Xva);
        oof(va,2,si) = sc(:,2);

        % Model 3: random forest, uniform class prior
        m3 = fitcensemble(Xtr, y(tr), 'Method','Bag', 'NumLearningCycles',200, ...
                          'Learners',templateTree('MinLeafSize',5), ...
                          'Prior','uniform', 'ClassNames',[0 1]);
        [~,sc] = predict(m3, Xva);
        oof(va,3,si) = sc(:,2);
    end
    fprintf('  seed %s done\n', seedNames{si});
end

save(fullfile(cfg.resDir,'baseline_oof.mat'), 'oof', 'y', 'modelNames');

% ---- Report: per-seed pooled AUC, then mean +/- SD across seeds
fprintf('\n================= RESULT =================\n');
fprintf('Lesions: %d (%d positive)   Seeds: %d x 5 folds\n\n', n, sum(y), nS);
fprintf('%-26s  AUC mean +/- SD   (per-seed AUCs)\n', 'Model');
for m = 1:nM
    a = zeros(1,nS);
    for si = 1:nS
        [~,~,~,a(si)] = perfcurve(y, oof(:,m,si), 1);
    end
    fprintf('%-26s  %.3f +/- %.3f     [%s]\n', modelNames{m}, mean(a), std(a), ...
            strtrim(sprintf('%.3f ', a)));
end
fprintf('==========================================\n');