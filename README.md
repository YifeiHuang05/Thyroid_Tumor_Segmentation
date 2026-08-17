# Thyroid_Tumor_Segmentation

Dataset Info from hunglc007/ThyroidXL from HuggingFace: <https://huggingface.co/datasets/hunglc007/ThyroidXL#thyroidxl-dataset>

# VSCode Setup

1. Create your own virtual environment using conda or pip.
2. run (ONLY IF YOU ARE ON WINDOWS!!!!!): pip install -r requirements.txt

# ThyroidXL Dataset

The ThyroidXL dataset. The annotations are provided for classification, detection and segmentation tasks.

Protocol:

Existing test set remains completely untouched. The existing training cohort is split once at the patient level using an 80:20 stratified split based on PTC status, with a fixed random seed. All images from the same patient are assigned to the same split. The validation set retains approximately the same PTC prevalence as the training cohort and is not oversampled. Class balancing, if required, is applied only during model training. The patient split is saved as a permanent manifest and reused for all subsequent experiments.



# Multi-task pipeline
Step 1 — Rebuild data
JSON
 ↓
patient-level split
 ↓
master CSV
Step 2 — Generate clean segmentation dataset
all polygons → class 0 = nodule
Step 3 — Train YOLO26n-seg

Use your existing training settings initially.

No HPO.

Step 4 — Calculate segmentation IoU

Use the actual polygon masks from the JSON as ground truth.

Step 5 — Generate classification crops

Use the ground-truth bounding boxes initially.

Step 6 — Train ResNet18 multi-task classifier

Four heads:

B/M
PTC
FNAC
TIRADS
Step 7 — Evaluate on the common test set

Produce:

Segmentation IoU
B/M Macro F1
PTC Macro F1
FNAC Macro F1
TIRADS Macro F1
Step 8 — Save everything
results.csv
confusion matrices
best.pt
classification_model.pt


# Segmentation
======================================================================
TEST SEGMENTATION RESULTS
======================================================================
Images evaluated:       2094
Mean IoU:               0.7979
Median IoU:             0.8328
Std IoU:                0.1490
Prediction presence:    0.9895
Mean inference time:    19.42 ms
Mean Dice coefficient:  0.8767
Mean HD95:              5.4922

Per-image results:
runs_clean_seg\thyroidxl_seg_clean\test_evaluation\segmentation_iou_per_image.csv

Summary:
runs_clean_seg\thyroidxl_seg_clean\test_evaluation\segmentation_iou_summary.csv



# Classification

======================================================================
PREPARING THYROIDXL CLASSIFICATION TARGETS
======================================================================

Input:
data\ThyroidXL_clean\metadata\master_annotations.csv

Loaded 11635 image records.

======================================================================
CLASSIFICATION LABEL AUDIT
======================================================================

Total images:   11635
Total patients: 4093

--- Benign / malignant ---
split  benign_malignant_name
test   benign                   1120
       malignant                 974
train  benign                   6018
       malignant                2115
val    benign                   1034
       malignant                 374
Name: count, dtype: int64

--- PTC / non-PTC ---
split  ptc_name
test   PTC          537
       non-PTC       41
       NaN         1516
train  PTC         1206
       non-PTC      177
       NaN         6750
val    PTC          230
       non-PTC       39
       NaN         1139
Name: count, dtype: int64

PTC label source:
split  ptc_source            
test   histopathology             578
       missing_histopathology    1516
train  histopathology            1383
       missing_histopathology    6750
val    histopathology             269
       missing_histopathology    1139
Name: count, dtype: int64

--- Official PTC test population ---
Test images with histopathology: 578
Test images without histopathology: 1516

Official PTC test labels:
ptc_name
PTC        537
non-PTC     41
Name: count, dtype: int64

--- FNAC ---
split  fnac_class
test   2.0           1126
       3.0             11
       4.0              7
       5.0             83
       6.0            768
       NaN             99
train  1.0             19
       2.0           5969
       3.0             88
       4.0             11
       5.0            195
       6.0           1641
       NaN            210
val    2.0           1038
       3.0              9
       5.0             30
       6.0            271
       NaN             60
Name: count, dtype: int64

--- TIRADS ---
split  tirads_class
test   2                131
       3                512
       4                617
       5                834
train  1                 64
       2                814
       3               2595
       4               2514
       5               2146
val    1                  6
       2                128
       3                455
       4                435
       5                384
Name: count, dtype: int64

--- Missing labels ---
benign_malignant         :     0 missing (0.00%)
ptc                      :  9405 missing (80.83%)
fnac_class               :   369 missing (3.17%)
tirads_class             :     0 missing (0.00%)

--- Patient-level consistency ---
benign_malignant         : 0 patients with inconsistent labels
ptc                      : 0 patients with inconsistent labels
fnac_class               : 0 patients with inconsistent labels
tirads_class             : 0 patients with inconsistent labels

======================================================================
CLASSIFICATION METADATA READY
======================================================================

Output directory:

data\ThyroidXL_clean\classification

Files:

classification_metadata.csv
patient_classification_labels.csv

benign_malignant_labels.csv
ptc_labels.csv
fnac_labels.csv
tirads_labels.csv





Epoch 17/25
  train loss = 2.0340
  val loss   = 2.7655
  time       = 87.7s
  validation Macro F1:
    benign_malignant     0.8000
    ptc                  0.8315
    fnac                 0.2636
    tirads               0.4823
  -> best classifier saved

Early stopping.

Classifier training time: 0.66 h

Loaded best classifier from epoch 17

======================================================================
GENERATING TEST CROPS FROM YOLO PREDICTED MASKS
======================================================================
Test crops saved to:
data\ThyroidXL_clean\classification\seg_guided_efficientnet\test_crops
YOLO segmentation failures: 22

======================================================================
OFFICIAL TEST CLASSIFICATION RESULTS
======================================================================

benign_malignant:
  Macro F1 = 0.8186
  Accuracy = 0.8195
  N = 2094

ptc:
  Macro F1 = 0.8031
  Accuracy = 0.9394
  N = 578

fnac:
  Macro F1 = 0.2590
  Accuracy = 0.6637
  N = 1995

tirads:
  Macro F1 = 0.4812
  Accuracy = 0.6428
  N = 2094

======================================================================
PIPELINE COMPLETE
======================================================================

Output directory:
data\ThyroidXL_clean\classification\seg_guided_efficientnet








# Two-stage thyroid pipeline foundation is in place
I implemented the dataset logic that matches your requested workflow:

A single patient-level split is created first, then reused for both stages.

Original labels are mapped as:
PTC → 0
NonPTC → 1
Benign → 2

Stage 1 remaps to:
Benign → 0
Malignant → 1

Stage 2 removes benign cases and remaps to:
PTC → 0
NonPTC → 1

Files added:

two_stage_dataset.py
test_two_stage_dataset.py

Generated dataset:

stage1.yaml
stage2.yaml


This implementation follows the required pattern:

Split once at patient level.
Reuse the same image/patient IDs for Stage 1 and Stage 2.
Relabel after splitting.
Keep Stage 2 restricted to malignant cases only.
Produce separate YOLO dataset YAMLs for each stage.

Stage 1: benign vs malignant YOLO segmentation
Stage 2: PTC vs NonPTC classification
Then apply the conditional fusion rule


## Dataset stats

Total number of patients: 4093 (patients)

Total number of images: 11545 (images)

### Train set

|         |  Train | Validation |
| ------- | -----: | ---------: |
| PTC     |   ~401 |       ~100 |
| Non-PTC | ~2,282 |       ~571 |
| Total   | ~2,683 |       ~671 |

train/val splitter:

split patients, not images;
stratify by PTC-positive vs PTC-negative patient;
use a fixed seed (2026);
create the split once and save it;
convert the JSON polygon annotations into YOLO segmentation .txt files;
copy images into images/train and images/val;
create labels/train and labels/val;
optionally populate images/test and labels/test from your existing test set;
save the original JSON subsets as well, which is useful for auditing;
generate dataset.yaml.

Number of patients: 3275 (2477 benigns, 877 malignants)

Number of images: 9541

Benign-Malignant Ratio: 64.6% - 35.4%

### Test set

Number of patients: 739 (386 benigns, 453 malignants)

Number of images: 2094

Benign-Malignant Ratio: 52.2% - 47.8%

## Category mapping

Gender:

Male: 1

Female: 2

Thyroid diagnostic results:

Benign: 0

Malignant: 1
