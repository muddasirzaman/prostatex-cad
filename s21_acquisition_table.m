%% s21_acquisition_table.m -- Acquisition parameters for the Methods section
clear; clc;
cfg = s00_config();
Str = load(fullfile(cfg.work,'series_index.mat'),'S');      Str = Str.S;
Ste = load(fullfile(cfg.work,'series_index_test.mat'),'S'); Ste = Ste.S;
Str.ProxID = string(Str.ProxID); Str.Sequence = string(Str.Sequence);
Ste.ProxID = string(Ste.ProxID); Ste.Sequence = string(Ste.Sequence);

load(fullfile(cfg.splitDir,'splits.mat'),'T_lesions');
trPat = unique(string(T_lesions.ProxID));
Te = readtable('D:\prostate cance\test_info\Findings-Test-merged.csv');
tePat = unique(string(Te.ProxID));

cohorts = {'Training', Str, trPat; 'Test', Ste, tePat};
t0 = tic;
for c = 1:2
    name = cohorts{c,1};  S = cohorts{c,2};  pats = cohorts{c,3};
    S = S(ismember(S.ProxID, pats), :);
    fprintf('\n================= %s COHORT =================\n', upper(name));
    fprintf('Patients with indexed series: %d\n', numel(unique(S.ProxID)));
    for q = {'t2','adc','dwi'}
        rows = find(S.Sequence == q{1});
        n = numel(rows);
        if n == 0, continue; end
        inpl = nan(n,1); thk = nan(n,1); tr = nan(n,1); te = nan(n,1); bval = nan(n,1);
        scan = strings(n,1); tesla = nan(n,1); mfr = strings(n,1);
        for k = 1:n
            f = dir(fullfile(S.Folder{rows(k)},'*.dcm'));
            if isempty(f), continue; end
            try, i1 = dicominfo(fullfile(f(1).folder, f(1).name)); catch, continue; end
            if isfield(i1,'PixelSpacing'),           inpl(k) = mean(double(i1.PixelSpacing)); end
            if isfield(i1,'SliceThickness'),         thk(k)  = double(i1.SliceThickness); end
            if isfield(i1,'RepetitionTime'),         tr(k)   = double(i1.RepetitionTime); end
            if isfield(i1,'EchoTime'),               te(k)   = double(i1.EchoTime); end
            if isfield(i1,'MagneticFieldStrength'),  tesla(k)= double(i1.MagneticFieldStrength); end
            if isfield(i1,'ManufacturerModelName'),  scan(k) = string(i1.ManufacturerModelName); end
            if isfield(i1,'Manufacturer'),           mfr(k)  = string(i1.Manufacturer); end
            for fld = ["DiffusionBValue","B_value"]
                if isfield(i1, fld), bval(k) = double(i1.(fld)); end
            end
        end
        fprintf('\n  %s  (%d series, %d patients)\n', upper(q{1}), n, numel(unique(S.ProxID(rows))));
        rep = @(lbl,v) fprintf('     %-18s median %.2f   range %.2f - %.2f   (missing %d)\n', ...
                               lbl, median(v,'omitnan'), min(v), max(v), sum(isnan(v)));
        rep('in-plane (mm)', inpl);  rep('slice thick (mm)', thk);
        rep('TR (ms)', tr);          rep('TE (ms)', te);
        if any(~isnan(bval)), rep('b-value (s/mm2)', bval); else, fprintf('     b-value          not in header\n'); end
        u = unique(scan(scan ~= ""));
        for j = 1:numel(u), fprintf('     scanner: %-12s %d series\n', u(j), sum(scan==u(j))); end
        u = unique(mfr(mfr ~= ""));
        fprintf('     manufacturer: %s | field strength: %s T\n', strjoin(cellstr(u'), ', '), ...
                strjoin(string(unique(tesla(~isnan(tesla)))'), ', '));
    end
end
fprintf('\nTotal time: %.1f minutes\n', toc(t0)/60);
fprintf('==========================================\n');