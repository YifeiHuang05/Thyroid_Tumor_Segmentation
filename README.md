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
