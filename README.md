# Thyroid_Tumor_Segmentation

Dataset Info from hunglc007/ThyroidXL from HuggingFace: <https://huggingface.co/datasets/hunglc007/ThyroidXL#thyroidxl-dataset>

# VSCode Setup

1. Create your own virtual environment using conda or pip.
2. run (ONLY IF YOU ARE ON WINDOWS!!!!!): pip install -r requirements.txt

# ThyroidXL Dataset

The ThyroidXL dataset. The annotations are provided for classification, detection and segmentation tasks.

Protocol:

Existing test set remains completely untouched. The existing training cohort is split once at the patient level using an 80:20 stratified split based on PTC status, with a fixed random seed. All images from the same patient are assigned to the same split. The validation set retains approximately the same PTC prevalence as the training cohort and is not oversampled. Class balancing, if required, is applied only during model training. The patient split is saved as a permanent manifest and reused for all subsequent experiments.

Patient-level split
        ↓
Train / Validation
        ↓
Training augmentation / sampling
        ↓
Model training

Define: PTC-positive =
    Papillary thyroid carcinoma
    + Micro-papillary thyroid carcinoma?
    + Papillary thyroid carcinoma, chronic thyroiditis?

PATIENT LEVEL
        │
        └── PTC status
              │
              ├── PTC+
              └── PTC−


(thyroidxl) C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation>python thyroidxl_pipeline.py    --source data\ThyroidXL     --output data\ThyroidXL_yolo    --prepare-only     

Source: C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL
Output: C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL_yolo

Loading JSON:
  C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL\train\train_annotations.json

Training patients: 3354
Training images: 9541
Training annotations: 9541

PTC-positive patients: 506
PTC-negative patients: 2848

No existing patient split found.
Creating a new stratified 80/20 split...

Saved permanent split:
  C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL_yolo\metadata\patient_split.csv
  C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL_yolo\metadata\train_patients.txt
  C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL_yolo\metadata\val_patients.txt

======================================================================
PATIENT-LEVEL SPLIT
======================================================================
TRAIN: 2683 patients |  7665 images |  405 PTC+ |  15.10% PTC
VAL:    671 patients |  1876 images |  101 PTC+ |  15.05% PTC
======================================================================

TRAIN: images=7665 polygons=7665 PTC=1164 non-PTC=6501 missing=0
VAL: images=1876 polygons=1876 PTC=272 non-PTC=1604 missing=0
TEST: images=2094 polygons=2094 PTC=537 non-PTC=1557 missing=0

Dataset YAML: C:\Users\250030817\Desktop\EEDP_EID_Project\Thyroid_Tumor_Segmentation\data\ThyroidXL_yolo\dataset.yaml
PTC patients: 506
non-PTC patients: 2848




0813 validating best.pt after epoch 54:
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95):
all       1876       1876      0.598      0.794      0.666      0.435        0.6      0.807      0.668      0.432
non-PTC       1604       1604      0.794      0.944      0.932      0.629      0.796      0.948      0.933      0.627
PTC        272        272      0.401      0.643      0.399      0.241      0.405      0.665      0.404      0.237








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
