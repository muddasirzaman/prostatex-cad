%% s18_train_final_models.m -- Train final models on ALL 330 training lesions (no test data used)
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'T_lesions');
n = height(T_lesions);  y = double(T_lesions.ClinSig);

Xf = zeros(n,12);
X  = zeros(cfg.patchPx, cfg.patchPx, 3, n, 'single');
for k = 1:n
    d = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    Xf(k,:) = makeFeatures(d.lesionPatches);
    t2  = single(d.lesionPatches.t2);
    adc = single(d.lesionPatches.adc);
    dwi = single(d.lesionPatches.dwi);
    X(:,:,1,k) = (t2  - mean(t2(:)))  / (std(t2(:))  + 1e-6);
    X(:,:,2,k) = adc;
    X(:,:,3,k) = (dwi - mean(dwi(:))) / (std(dwi(:)) + 1e-6);
end
fprintf('Training on %d lesions (%d positive)\n', n, sum(y));
t0 = tic;

% ---- Logistic regression and random forest (same settings as s13)
mu = mean(Xf);  sd = std(Xf) + 1e-9;
Xs = (Xf - mu) ./ sd;
rng(0);
m2 = fitclinear(Xs, y, 'Learner','logistic', 'Regularization','ridge', ...
                'Lambda',0.05, 'Prior','uniform', 'ClassNames',[0 1]);
m3 = fitcensemble(Xs, y, 'Method','Bag', 'NumLearningCycles',200, ...
                  'Learners',templateTree('MinLeafSize',5), ...
                  'Prior','uniform', 'ClassNames',[0 1]);
fprintf('Logistic and random forest done (%.1f min)\n', toc(t0)/60);

% ---- Small CNN (same settings as s14), 5 seeds
a = X(:,:,2,:);  adcMu = mean(a(:));  adcSd = std(a(:)) + 1e-6;
X(:,:,2,:) = (X(:,:,2,:) - adcMu) / adcSd;
w = n ./ (2 * [sum(y==0), sum(y==1)]);
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
opts = trainingOptions('adam', 'MaxEpochs',30, 'MiniBatchSize',16, ...
    'InitialLearnRate',1e-3, 'L2Regularization',1e-3, ...
    'Shuffle','every-epoch', 'Verbose',false, 'Plots','none', 'ExecutionEnvironment','cpu');
aug = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true, ...
                         'RandRotation',[-20 20], ...
                         'RandXTranslation',[-3 3], 'RandYTranslation',[-3 3]);
Ytr = categorical(y);
nets = cell(1,5);
for s = 1:5
    rng(1000*s);
    ds = augmentedImageDatastore([cfg.patchPx cfg.patchPx 3], X, Ytr, 'DataAugmentation', aug);
    nets{s} = trainNetwork(ds, layers, opts);
    fprintf('  CNN seed %d done   (%.1f min elapsed)\n', s, toc(t0)/60);
end

save(fullfile(cfg.modelDir,'final_models.mat'), 'm2','m3','mu','sd','nets','adcMu','adcSd','-v7.3');
fprintf('\n================= RESULT =================\n');
fprintf('Final models trained on %d lesions and saved to:\n  %s\n', n, fullfile(cfg.modelDir,'final_models.mat'));
fprintf('Total time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');

%% ---- helper (identical to the features used in s13)
function f = makeFeatures(lp)
seq = {'t2','adc','dwi'};  f = zeros(1,12);
for s = 1:3
    im  = double(lp.(seq{s}));
    c1  = im(25:40,25:40);  c2 = im(17:48,17:48);  med = median(im(:));
    f((s-1)*4 + (1:4)) = [mean(c1(:)), mean(c2(:)), std(c2(:)), mean(c1(:)) / (abs(med) + 1e-6)];
end
end