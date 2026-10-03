%% s16_export_for_python.m -- Corrected patches + the same 25 splits, for PyTorch
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'splits','T_lesions');
n = height(T_lesions);
y = double(T_lesions.ClinSig);

X = zeros(cfg.patchPx, cfg.patchPx, 3, n, 'single');
for k = 1:n
    d   = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    t2  = single(d.lesionPatches.t2);
    adc = single(d.lesionPatches.adc);
    dwi = single(d.lesionPatches.dwi);
    X(:,:,1,k) = (t2  - mean(t2(:)))  / (std(t2(:))  + 1e-6);   % per-patch z-score
    X(:,:,2,k) = adc;                                            % raw; standardised per fold in Python
    X(:,:,3,k) = (dwi - mean(dwi(:))) / (std(dwi(:)) + 1e-6);   % per-patch z-score
end

seedNames = fieldnames(splits);
nS = numel(seedNames);
foldOf = zeros(n, nS);                 % foldOf(i,s) = fold in which lesion i is VALIDATION
for si = 1:nS
    for fi = 1:cfg.nFolds
        va = splits.(seedNames{si}).(sprintf('fold_%d',fi)).valIdx(:);
        foldOf(va,si) = fi;
    end
end
assert(all(foldOf(:) > 0), 'Some lesions never validated');

outFile = fullfile(cfg.work,'dataset_python.mat');
save(outFile, 'X', 'y', 'foldOf', '-v7');
fprintf('Saved %s\n', outFile);
fprintf('X: %s   lesions: %d (%d positive)   seeds: %d\n', mat2str(size(X)), n, sum(y), nS);