function cfg = s00_config()
%% ========================================================================
%  s00_config.m -- ALL settings for the project, in ONE place.
%  Every other script begins with:   cfg = s00_config();
%% ========================================================================

% ------------------------- WHERE THINGS ARE -----------------------------
cfg.base        = 'D:\prostate cance';
cfg.dicomRoot   = fullfile(cfg.base,'dicom');
cfg.labelDir    = fullfile(cfg.base,'ProstateX-TrainingLesionInformationv2');
cfg.shotsDir    = fullfile(cfg.base,'ProstateX-Screenshots-Train');

cfg.findingsCSV = fullfile(cfg.labelDir,'ProstateX-Findings-Train.csv');
cfg.imagesCSV   = fullfile(cfg.labelDir,'ProstateX-Images-Train.csv');

% ------------------- WHERE RESULTS WILL BE SAVED ------------------------
cfg.work     = fullfile(cfg.base,'study');
cfg.patchDir = fullfile(cfg.work,'patches');
cfg.splitDir = fullfile(cfg.work,'splits');
cfg.ckptDir  = fullfile(cfg.work,'checkpoints');
cfg.modelDir = fullfile(cfg.work,'models');
cfg.resDir   = fullfile(cfg.work,'results');
cfg.figDir   = fullfile(cfg.work,'figures');

outs = {cfg.work,cfg.patchDir,cfg.splitDir,cfg.ckptDir, ...
        cfg.modelDir,cfg.resDir,cfg.figDir};
for k = 1:numel(outs)
    if ~exist(outs{k},'dir'); mkdir(outs{k}); end
end

% ------------------------ SCIENTIFIC SETTINGS ---------------------------
cfg.sequences = {'t2','adc','dwi'};
cfg.patchMM   = 24;
cfg.patchPx   = 64;
cfg.netInput  = [227 227];
cfg.nFolds    = 5;
cfg.seeds     = [1 2 3 4 5];

% --------------------- HOW TO RECOGNISE EACH SEQUENCE -------------------
% ORDER MATTERS. "_ADC" and "CALC_BVAL" both contain "DYNDIST".
% Most specific pattern first, or ADC and DWI get swapped.
cfg.pat.adc = {'DYNDIST_MIX_ADC','DYNDIST_ADC','4bval_fs_ADC'};
cfg.pat.dwi = {'DYNDIST_MIXCALC_BVAL','DYNDISTCALC_BVAL','4bval_fsCALC_BVAL'};
cfg.pat.t2  = {'t2_tse_tra'};

cfg.patExclude = {'sag','cor','localizer','_loc','tfl_dyn','PD ref','_PD_'};

end