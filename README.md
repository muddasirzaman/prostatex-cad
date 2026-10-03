# Lesion-level classification of clinically significant prostate cancer on PROSTATEx

Code for the study *"A compact convolutional network as a strong baseline for lesion-level
classification of clinically significant prostate cancer on multiparametric MRI"*.

**[TO DO: replace the title above if it changes, and add the citation once the paper is published.]**

The pipeline extracts lesion-centred multiparametric patches from the public PROSTATEx
collection, compares a fixed ADC rule, two feature-based classifiers, a compact CNN and a
fine-tuned ResNet-18, and evaluates the final models once on the official PROSTATEx test cohort.

## Results summary

Independent test cohort, 206 of 208 lesions from 139 patients (47 clinically significant).
Areas under the ROC curve with 95% confidence intervals from patient-level bootstrap:

| Model | AUC | 95% CI |
|---|---|---|
| Compact CNN (5-network average) | 0.869 | 0.811 – 0.921 |
| ADC rule (no training) | 0.799 | 0.723 – 0.861 |
| Random forest (12 features) | 0.787 | 0.709 – 0.854 |
| Logistic regression (12 features) | 0.773 | 0.697 – 0.845 |

Cross-validation on the training cohort (5 folds x 5 seeds, patient-level): compact CNN 0.797,
random forest 0.765, logistic regression 0.752, ADC rule 0.740, fine-tuned ResNet-18 0.716.

## Data

The PROSTATEx collection is **not** included in this repository. Download it from The Cancer
Imaging Archive:

- Imaging and lesion information: https://www.cancerimagingarchive.net/collection/prostatex/
- Licence: CC BY 3.0

Data citation:

> Litjens, G., Debats, O., Barentsz, J., Karssemeijer, N., & Huisman, H. (2017).
> SPIE-AAPM PROSTATEx Challenge Data (Version 2) [dataset]. The Cancer Imaging Archive.
> https://doi.org/10.7937/K9TCIA.2017.MURS5CL

Expected local layout (set the base path in `s00_config.m`):

```
D:\prostate cance\
    dicom\                                  ProstateX-#### folders
    ProstateX-TrainingLesionInformationv2\  training findings and image CSVs
    test_info\                              test findings CSV and the published reference standard
    study\                                  created automatically by the code
    code\                                   the scripts in this repository
```

## Requirements

- MATLAB R2021a with the Deep Learning Toolbox, the Statistics and Machine Learning Toolbox
  and the Image Processing Toolbox
- Python 3.14 with `torch` 2.14, `torchvision` 0.29, `scipy`, `scikit-learn` and `numpy`,
  for the pretrained network only

No GPU is required. Every experiment in the paper was run on a CPU.

## How to run

Set `cfg.base` in `s00_config.m` to your data folder, then run the scripts in this order.
Approximate run times are for a laptop CPU.

| Step | Script | Purpose | Time |
|---|---|---|---|
| 1 | `s02_index_dicom.m` | index T2, ADC and DWI series for the training cohort | 15 min |
| 2 | `s03_extract_patches.m` | extract 24 mm lesion patches | 10 min |
| 3 | `s04_create_splits.m` | patient-level 5-fold x 5-seed partitions | seconds |
| 4 | `s13_baseline_all_splits.m` | ADC rule, logistic regression, random forest | 5 min |
| 5 | `s14_cnn_all_splits.m` | compact CNN over all 25 partitions | 22 min |
| 6 | `s16_export_for_python.m` | export patches and partitions for PyTorch | 1 min |
| 7 | `train_resnet_cv.py 5` | fine-tuned ResNet-18 over the same partitions | 150 min |
| 8 | `s03b_extract_context_patches.m` | 48 mm and 72 mm patches (run twice) | 35 min |
| 9 | `s14b_cnn_context.m` | CNN on the larger fields of view (run twice) | 70 min |
| 10 | `s23_imbalance_ablation.m` | no weighting and oversampling variants | 62 min |
| 11 | `s15b_statistics_all.m` | bootstrap intervals, paired tests, ROC | 2 min |
| 12 | `s15c_context_stats.m` | field-of-view comparison | 2 min |
| 13 | `s17b_merge_test_files.m` | merge official test coordinates with the reference standard | seconds |
| 14 | `s17_extract_test_patches.m` | index and extract test patches | 36 min |
| 15 | `s18_train_final_models.m` | train final models on all 330 training lesions | 8 min |
| 16 | `s19_evaluate_test.m` | **single** evaluation on the test cohort | 3 min |
| 17 | `s20_test_metrics.m` | operating points, confusion matrices, PR curves | 2 min |
| 18 | `s21_acquisition_table.m` | acquisition parameters | 2 min |
| 19 | `s22_gradcam.m` | class activation maps | 1 min |
| 20 | `s24_figure_examples.m` | example patch figure | seconds |

Step 16 is intended to be run once. Every model, preprocessing and threshold choice in this
pipeline was fixed before it was run.

## Results files

The `results/` folder holds the three summary tables reported in the paper, so the numbers can
be inspected without running the pipeline:

- `auc_table_all.csv` — cross-validation AUC with bootstrap intervals for all models
- `auc_table_context.csv` — the same with the 48 mm and 72 mm field-of-view variants
- `auc_table_test.csv` — independent test cohort AUC with bootstrap intervals

## Notes on the dataset

Three issues were encountered that may affect anyone else using this collection:

1. **Duplicate finding identifiers.** Three training patients (ProstateX-0005, -0025, -0159)
   have two distinct findings sharing `fid = 1`. Writing patch files keyed on patient and finding
   identifier silently overwrites one of each pair.
2. **Non-uniform diffusion series naming.** Besides the common `ep2d_diff_tra_DYNDIST` family,
   subsets of the collection use `diffusie-3Scan-4bval_fs`, `diff tra b 50 500 800 WIP511b` and
   `ep2d-advdiff-3Scan-4bval_spair_511b`. Matching only the first family loses many patients.
3. **Test coordinates.** The published test reference standard contains truncated coordinates for
   26 of 208 lesions. Full coordinates are in the test lesion information package; `s17b` merges
   the two and verifies that they agree.

## Licence

Code released under the MIT licence (see `LICENSE`). The PROSTATEx data is licensed separately
by The Cancer Imaging Archive under CC BY 3.0.

## Contact

Muddasir Zaman, Department of Biomedical Engineering, Salim Habib University, Karachi.
muddasirzaman799@gmail.com
