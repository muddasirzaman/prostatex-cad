%% s17b_merge_test_files.m -- Official test coordinates + labels, with consistency checks
clear; clc;
coordFile = 'C:\Users\HP\Desktop\PKG - PROSTATEx\ProstateX-TestLesionInformation\ProstateX-Findings-Test.csv';
labFile   = 'D:\prostate cance\test_info\ProstateX-Findings-Test.csv';   % the file with ClinSig
outFile   = 'D:\prostate cance\test_info\Findings-Test-merged.csv';

copyfile(coordFile, 'D:\prostate cance\test_info\Findings-Test-official.csv');   % keep a record
A = readtable(coordFile);      % official coordinates
B = readtable(labFile);        % labels file
fprintf('Official file columns : %s\n', strjoin(A.Properties.VariableNames, ', '));
fprintf('Labels file columns   : %s\n\n', strjoin(B.Properties.VariableNames, ', '));

parse3 = @(v) sscanf(strrep(strrep(strrep(char(string(v)),'[',' '),']',' '),',',' '),'%f')';

kA = strcat(string(A.ProxID),"_",string(A.fid));
kB = strcat(string(B.ProxID),"_",string(B.fid));
fprintf('Rows: official %d | labels file %d\n', height(A), height(B));
fprintf('Duplicate keys: official %d | labels file %d\n', numel(kA)-numel(unique(kA)), numel(kB)-numel(unique(kB)));
fprintf('Keys only in official: %d | only in labels file: %d\n', sum(~ismember(kA,kB)), sum(~ismember(kB,kA)));
[~, ia, ib] = intersect(kA, kB, 'stable');
fprintf('Matched lesions: %d\n\n', numel(ia));

% zone agreement
zoneOK = strcmpi(strtrim(string(A.zone(ia))), strtrim(string(B.zone(ib))));
fprintf('Zone agrees: %d of %d\n', sum(zoneOK), numel(ia));
if any(~zoneOK)
    disp(table(kA(ia(~zoneOK)), string(A.zone(ia(~zoneOK))), string(B.zone(ib(~zoneOK))), 'VariableNames',{'Key','OfficialZone','LabelsZone'}));
end

% coordinate parsing and agreement
nBadA = 0; nCmp = 0; maxD = 0; nFar = 0;
for i = 1:numel(ia)
    pa = parse3(A.pos(ia(i)));
    if numel(pa) ~= 3, nBadA = nBadA + 1; continue; end
    pb = parse3(B.pos(ib(i)));
    if numel(pb) == 3
        d = norm(pa - pb); nCmp = nCmp + 1; maxD = max(maxD, d); nFar = nFar + (d > 0.01);
    end
end
fprintf('Official positions that do NOT parse to 3 numbers: %d\n', nBadA);
fprintf('Positions compared against labels file: %d | max difference %.4f mm | differing by >0.01 mm: %d\n\n', nCmp, maxD, nFar);

% merged file: coordinates + zone from official, ClinSig from labels file
M = table(A.ProxID(ia), A.fid(ia), A.pos(ia), A.zone(ia), B.ClinSig(ib), ...
          'VariableNames', {'ProxID','fid','pos','zone','ClinSig'});
writetable(M, outFile);
isPos = strcmpi(strtrim(string(M.ClinSig)),"true");
fprintf('================= RESULT =================\n');
fprintf('Merged lesions : %d (%d positive, %d negative) from %d patients\n', height(M), sum(isPos), sum(~isPos), numel(unique(string(M.ProxID))));
fprintf('Saved to       : %s\n', outFile);
fprintf('==========================================\n');