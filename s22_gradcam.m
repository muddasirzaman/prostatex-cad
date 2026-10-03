%% s22_gradcam.m -- Class activation maps for the final CNN ensemble (test set)
clear; clc;
cfg = s00_config();
M  = load(fullfile(cfg.modelDir,'final_models.mat'));     % nets, adcMu, adcSd
S  = load(fullfile(cfg.resDir,'test_scores.mat'));        % sc, y, pid, names
Mt = load(fullfile(cfg.resDir,'test_metrics.mat'));       % thr

files = dir(fullfile(cfg.work,'patches_test','test_*.mat'));
n = numel(files);
X = zeros(cfg.patchPx, cfg.patchPx, 3, n, 'single');
T2 = zeros(cfg.patchPx, cfg.patchPx, n);
y  = zeros(n,1);
for k = 1:n
    d = load(fullfile(files(k).folder, files(k).name));
    t2  = single(d.lesionPatches.t2);
    adc = single(d.lesionPatches.adc);
    dwi = single(d.lesionPatches.dwi);
    X(:,:,1,k) = (t2  - mean(t2(:)))  / (std(t2(:))  + 1e-6);
    X(:,:,2,k) = (adc - M.adcMu) / M.adcSd;
    X(:,:,3,k) = (dwi - mean(dwi(:))) / (std(dwi(:)) + 1e-6);
    T2(:,:,k)  = double(t2);
    y(k) = d.label;
end
assert(isequal(y, S.y(:)), 'Patch order does not match test_scores.mat');

% ---- Locate the CAM layer and the class weights
L  = M.nets{1}.Layers;
ri = find(arrayfun(@(l) isa(l,'nnet.cnn.layer.ReLULayer'), L));
fi = find(arrayfun(@(l) isa(l,'nnet.cnn.layer.FullyConnectedLayer'), L));
camLayer = L(ri(end)).Name;
cls = string(L(end).Classes);
posIdx = find(cls == "1");
fprintf('CAM layer: %s | positive class index: %d\n', camLayer, posIdx);

% ---- CAM averaged over the 5 networks
camSum = [];
for s = 1:numel(M.nets)
    A = activations(M.nets{s}, X, camLayer);              % H x W x C x N
    w = M.nets{s}.Layers(fi(end)).Weights(posIdx,:);      % 1 x C
    c = squeeze(sum(A .* reshape(w,1,1,[]), 3));          % H x W x N
    if isempty(camSum), camSum = c; else, camSum = camSum + c; end
end
camSum = camSum / numel(M.nets);

cam = zeros(cfg.patchPx, cfg.patchPx, n);
for k = 1:n
    c = max(camSum(:,:,k), 0);
    c = imresize(c, [cfg.patchPx cfg.patchPx], 'bilinear');
    if max(c(:)) > 0, c = c / max(c(:)); end
    cam(:,:,k) = c;
end

% ---- Quantitative: where does the attention sit?
mmPerPx = cfg.patchMM / cfg.patchPx;
ctr = (cfg.patchPx + 1) / 2;
[gx, gy] = meshgrid(1:cfg.patchPx, 1:cfg.patchPx);
box = abs(gx-ctr) <= 12 & abs(gy-ctr) <= 12;              % central 9 mm box
dist = zeros(n,1); frac = zeros(n,1);
for k = 1:n
    c = cam(:,:,k);  tot = sum(c(:)) + 1e-9;
    dist(k) = hypot(sum(gx(:).*c(:))/tot - ctr, sum(gy(:).*c(:))/tot - ctr) * mmPerPx;
    frac(k) = sum(c(box)) / tot;
end

thrB = Mt.thr(4,2);
pred = S.sc(:,4) >= thrB;
grp  = strings(n,1);
grp(y==1 &  pred) = "TP";  grp(y==0 & ~pred) = "TN";
grp(y==0 &  pred) = "FP";  grp(y==1 & ~pred) = "FN";

fprintf('\n================= RESULT =================\n');
fprintf('Class activation maps, %d test lesions, CNN ensemble, Youden threshold\n', n);
fprintf('Chance level for the central-box fraction: %.3f (box area / patch area)\n\n', sum(box(:))/numel(box));
fprintf('%-5s %5s   centroid distance from centre (mm)   fraction of attention in central 9 mm\n','Group','n');
for g = ["TP","FP","FN","TN"]
    i = grp == g;
    if ~any(i), continue; end
    fprintf('%-5s %5d            %.2f +/- %.2f                        %.3f +/- %.3f\n', ...
        g, sum(i), mean(dist(i)), std(dist(i)), mean(frac(i)), std(frac(i)));
end
fprintf('\nAll lesions: distance %.2f +/- %.2f mm | central fraction %.3f +/- %.3f\n', ...
        mean(dist), std(dist), mean(frac), std(frac));
fprintf('(Descriptive only; patches are lesion-centred by construction)\n');
fprintf('==========================================\n');

% ---- Figure: two examples per group, T2 with CAM overlay
rng(7);
pick = [];
for g = ["TP","FP","FN","TN"]
    i = find(grp == g);
    pick = [pick; i(randperm(numel(i), min(2,numel(i))))]; %#ok<AGROW>
end
figure('Color','w','Position',[80 80 1100 580]);
for j = 1:numel(pick)
    k = pick(j);
    base = ind2rgb(gray2ind(mat2gray(T2(:,:,k)),256), gray(256));
    heat = ind2rgb(uint8(255*cam(:,:,k)), jet(256));
    subplot(2,4,j);
    imshow(0.55*base + 0.45*heat); hold on;
    plot(ctr, ctr, 'w+', 'MarkerSize', 10, 'LineWidth', 1.2);
    title(sprintf('%s  score %.2f', grp(k), S.sc(k,4)), 'FontSize', 9);
end
sgtitle('CNN attention on test lesions (T2 with class activation map; + = lesion centre)');
print(gcf, fullfile(cfg.figDir,'gradcam_test.png'), '-dpng', '-r300');
save(fullfile(cfg.resDir,'cam_results.mat'), 'cam','dist','frac','grp','y');
fprintf('Saved: gradcam_test.png and cam_results.mat\n');