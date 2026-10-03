%% s23_imbalance_ablation.m -- Does imbalance handling change the CNN result?
%  Variant 1 (class weights) is already done: cnn_oof.mat, AUC 0.797.
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'splits','T_lesions');
n = height(T_lesions);  y = double(T_lesions.ClinSig);

X = zeros(cfg.patchPx, cfg.patchPx, 3, n, 'single');
for k = 1:n
    d = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    t2  = single(d.lesionPatches.t2);
    adc = single(d.lesionPatches.adc);
    dwi = single(d.lesionPatches.dwi);
    X(:,:,1,k) = (t2  - mean(t2(:)))  / (std(t2(:))  + 1e-6);
    X(:,:,2,k) = adc;
    X(:,:,3,k) = (dwi - mean(dwi(:))) / (std(dwi(:)) + 1e-6);
end
seedNames = fieldnames(splits);
aug = imageDataAugmenter('RandXReflection',true,'RandYReflection',true, ...
        'RandRotation',[-20 20],'RandXTranslation',[-3 3],'RandYTranslation',[-3 3]);
modes = {'noweight','oversample'};
t0 = tic;

for m = 1:2
    mode = modes{m};
    oof = nan(n,5);
    for si = 1:5
        for fi = 1:5
            sp = splits.(seedNames{si}).(sprintf('fold_%d',fi));
            tr = sp.trainIdx(:);  va = sp.valIdx(:);
            Xtr = X(:,:,:,tr);    Xva = X(:,:,:,va);
            a = Xtr(:,:,2,:);  mu = mean(a(:));  sd = std(a(:)) + 1e-6;
            Xtr(:,:,2,:) = (Xtr(:,:,2,:) - mu)/sd;
            Xva(:,:,2,:) = (Xva(:,:,2,:) - mu)/sd;
            ytr = y(tr);

            rng(si*100 + fi);
            if strcmp(mode,'oversample')
                p = find(ytr==1);  q = find(ytr==0);
                rep = repmat(p, ceil(numel(q)/numel(p)), 1);
                sel = [q; rep(1:numel(q))];
                sel = sel(randperm(numel(sel)));
                Xtr = Xtr(:,:,:,sel);  ytr = ytr(sel);
            end
            Ytr = categorical(ytr);

                        layers = [
                imageInputLayer([cfg.patchPx cfg.patchPx 3],'Normalization','none')
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
                classificationLayer('Classes',categorical([0 1]))   % no class weights
            ];
            opts = trainingOptions('adam','MaxEpochs',30,'MiniBatchSize',16, ...
                'InitialLearnRate',1e-3,'L2Regularization',1e-3, ...
                'Shuffle','every-epoch','Verbose',false,'Plots','none', ...
                'ExecutionEnvironment','cpu');
            ds  = augmentedImageDatastore([cfg.patchPx cfg.patchPx 3], Xtr, Ytr, 'DataAugmentation', aug);
            net = trainNetwork(ds, layers, opts);
            sc  = predict(net, Xva);
            oof(va,si) = sc(:,2);
        end
        fprintf('  %s: seed %d done (%.1f min)\n', mode, si, toc(t0)/60);
    end
    a = zeros(1,5);
    for si = 1:5, [~,~,~,a(si)] = perfcurve(y, oof(:,si), 1); end
    fprintf('>> %s: AUC %.3f +/- %.3f\n', mode, mean(a), std(a));
    save(fullfile(cfg.resDir, sprintf('cnn_oof_%s.mat', mode)), 'oof','y','a');
end

fprintf('\n================= RESULT =================\n');
fprintf('Imbalance handling, same CNN, 25 splits, 330 lesions (76 positive)\n');
fprintf('  class weights (reference) : 0.797\n');
for m = 1:2
    L = load(fullfile(cfg.resDir, sprintf('cnn_oof_%s.mat', modes{m})));
    fprintf('  %-24s : %.3f +/- %.3f\n', modes{m}, mean(L.a), std(L.a));
end
fprintf('Total time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');