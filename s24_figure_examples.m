%% s24_figure_examples.m -- Figure 2: example lesion patches (training cohort)
clear; clc;
cfg = s00_config();
load(fullfile(cfg.splitDir,'splits.mat'),'T_lesions');
n = height(T_lesions);  y = double(T_lesions.ClinSig);

% Central 6 mm mean ADC for every lesion, used only to choose representative cases
c = 25:40;
adcC = zeros(n,1);
for k = 1:n
    d = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    adcC(k) = mean2(double(d.lesionPatches.adc(c,c)));
end

% Pick the MEDIAN case in each class (representative, not best case)
pickMedian = @(idx) idx(find(adcC(idx) == median(adcC(idx)),1));
pos = find(y==1);  neg = find(y==0);
[~,ip] = min(abs(adcC(pos) - median(adcC(pos))));  kPos = pos(ip);
[~,inn] = min(abs(adcC(neg) - median(adcC(neg))));  kNeg = neg(inn);
sel = [kPos kNeg];
lab = {'Clinically significant','Not clinically significant'};

% Shared ADC window across both rows (ADC is quantitative)
A = [];
for k = sel
    d = load(fullfile(cfg.patchDir, T_lesions.FileName{k}));
    A = [A; double(d.lesionPatches.adc(:))]; %#ok<AGROW>
end
adcLim = [prctile(A,1) prctile(A,99)];

mmPerPx = cfg.patchMM / cfg.patchPx;
barPx   = 10 / mmPerPx;                       % 10 mm scale bar
seqName = {'T2-weighted','ADC','DWI (high b)'};
seqKey  = {'t2','adc','dwi'};

figure('Color','w','Position',[80 80 760 560]);
t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
for r = 1:2
    d = load(fullfile(cfg.patchDir, T_lesions.FileName{sel(r)}));
    fprintf('Row %d (%s): %s, finding %d\n', r, lab{r}, ...
            string(T_lesions.ProxID{sel(r)}), T_lesions.LesionID(sel(r)));
    for s = 1:3
        im = double(d.lesionPatches.(seqKey{s}));
        nexttile;
        if s == 2
            imshow(im, adcLim);                 % shared quantitative window
        else
            imshow(im, []);                     % per-image window (arbitrary units)
        end
        hold on;
        plot(32.5, 32.5, 'r+', 'MarkerSize', 11, 'LineWidth', 1.3);
        if r == 1, title(seqName{s}, 'FontSize', 10); end
        if s == 1
            ylabel(lab{r}, 'Visible','on', 'FontSize', 9, 'FontWeight','bold');
            set(gca,'YTick',[],'XTick',[],'Visible','on','Box','off', ...
                    'XColor','none','YColor','k');
        end
        if r == 2 && s == 3                      % scale bar on one panel only
            x0 = cfg.patchPx - barPx - 4;  y0 = cfg.patchPx - 5;
            plot([x0 x0+barPx], [y0 y0], 'w-', 'LineWidth', 3);
            text(x0 + barPx/2, y0 - 4, '10 mm', 'Color','w', ...
                 'HorizontalAlignment','center', 'FontSize', 8);
        end
    end
end
title(t, 'Lesion-centred 24 mm patches on a 64 \times 64 grid (+ = annotated lesion centre)', ...
      'FontSize', 10);
print(gcf, fullfile(cfg.figDir,'fig2_example_patches.png'), '-dpng', '-r300');
fprintf('Saved: %s\n', fullfile(cfg.figDir,'fig2_example_patches.png'));