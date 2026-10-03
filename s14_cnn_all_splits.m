%% s14_cnn_all_splits.m -- Small CNN over all 5 folds x 5 seeds (CPU)
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'splits','T_lesions');

nSeedsToRun = 5;     % START WITH 1 (about 15 min). After it works, set to 5 (about 1.5 h)
nEpochs     = 30;    % fixed; no epoch picking on validation data

n = height(T_lesions);
y = double(T_lesions.ClinSig);
X = zeros(cfg.patchPx, cfg.patchPx, 3, n, 'single');
for k = 1:n
    d   = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    t2  = single(d.lesionPatches.t2);
    adc = single(d.lesionPatches.adc);
    dwi = single(d.lesionPatches.dwi);
    X(:,:,1,k) = (t2  - mean(t2(:)))  / (std(t2(:))  + 1e-6);   % per-patch z-score
    X(:,:,2,k) = adc;                                            % raw; standardised per fold below
    X(:,:,3,k) = (dwi - mean(dwi(:))) / (std(dwi(:)) + 1e-6);   % per-patch z-score
end

seedNames = fieldnames(splits);
nS = min(nSeedsToRun, numel(seedNames));
oof = nan(n, nS);
t0 = tic;

aug = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true, ...
                         'RandRotation',[-20 20], ...
                         'RandXTranslation',[-3 3], 'RandYTranslation',[-3 3]);

for si = 1:nS
    foldNames = fieldnames(splits.(seedNames{si}));
    for fi = 1:numel(foldNames)
        sp = splits.(seedNames{si}).(foldNames{fi});
        tr = sp.trainIdx(:);  va = sp.valIdx(:);
        Xtr = X(:,:,:,tr);    Xva = X(:,:,:,va);

        % ADC standardised with TRAIN-fold statistics only
        a  = Xtr(:,:,2,:);  mu = mean(a(:));  sd = std(a(:)) + 1e-6;
        Xtr(:,:,2,:) = (Xtr(:,:,2,:) - mu) / sd;
        Xva(:,:,2,:) = (Xva(:,:,2,:) - mu) / sd;

        Ytr = categorical(y(tr));
        w   = numel(tr) ./ (2 * [sum(y(tr)==0), sum(y(tr)==1)]);   % class weights

        layers = [
            imageInputLayer([cfg.patchPx cfg.patchPx 3], 'Normalization','none')
            convolution2dLayer(3,16,'Padding','same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(3,32,'Padding','same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(3,64,'Padding','same')
            batchNormalizationLayer
            reluLayer
            globalAveragePooling2dLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer('Classes',categorical([0 1]), 'ClassWeights',w)
        ];

        opts = trainingOptions('adam', 'MaxEpochs',nEpochs, 'MiniBatchSize',16, ...
            'InitialLearnRate',1e-3, 'L2Regularization',1e-3, ...
            'Shuffle','every-epoch', 'Verbose',false, 'Plots','none', ...
            'ExecutionEnvironment','cpu');

        rng(si*100 + fi);
        ds  = augmentedImageDatastore([cfg.patchPx cfg.patchPx 3], Xtr, Ytr, 'DataAugmentation', aug);
        net = trainNetwork(ds, layers, opts);

        sc = predict(net, Xva);          % N x 2 probabilities; column 2 = class '1'
        oof(va,si) = sc(:,2);
        fprintf('  %s %s done   (%.1f min elapsed)\n', seedNames{si}, foldNames{fi}, toc(t0)/60);
    end
end

fprintf('\n================= RESULT =================\n');
fprintf('Small CNN, %d seed(s) x 5 folds, %d epochs, %d lesions (%d positive)\n', nS, nEpochs, n, sum(y));
a = zeros(1,nS);
for si = 1:nS
    [~,~,~,a(si)] = perfcurve(y, oof(:,si), 1);
end
fprintf('CNN pooled AUC : %.3f +/- %.3f   [%s]\n', mean(a), std(a), strtrim(sprintf('%.3f ', a)));
fprintf('Reference      : ADC-only 0.740 | Logistic 0.752 | Random forest 0.765\n');
fprintf('Total time     : %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');
save(fullfile(cfg.resDir,'cnn_oof.mat'), 'oof', 'y', 'a');