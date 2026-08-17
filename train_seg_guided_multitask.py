from pathlib import Path
import copy
import json
import time
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    "data/ThyroidXL_clean"
)

METADATA = (
    ROOT
    / "classification"
    / "classification_metadata.csv"
)

SEG_MODEL = Path(
    "runs/segment/runs_clean_seg/thyroidxl_seg_clean/weights/best.pt"
)

IMAGE_ROOT = (
    ROOT
    / "segmentation"
    / "images"
)

LABEL_ROOT = (
    ROOT
    / "segmentation"
    / "labels"
)

OUTPUT = (
    ROOT
    / "classification"
    / "seg_guided_efficientnet"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
EPOCHS = 25
PATIENCE = 5
IMAGE_SIZE = 224

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

SEED = 42

# Percentage of extra context around the nodule bounding box.
# 0.20 = 20% padding on each side.
CROP_MARGIN = 0.20

# YOLO confidence threshold for the official test pipeline.
CONF = 0.25


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# YOLO GROUND-TRUTH MASK
# ============================================================

def yolo_label_to_mask(
    label_path,
    width,
    height
):
    """
    Convert YOLO segmentation polygons into a binary mask.

    Clean dataset:
        class 0 = thyroid nodule
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if not label_path.exists():
        return mask

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 7:
                continue

            class_id = int(
                values[0]
            )

            if class_id != 0:
                continue

            coords = np.asarray(
                [
                    float(x)
                    for x in values[1:]
                ],
                dtype=np.float32
            )

            if len(coords) < 6:
                continue

            points = coords.reshape(
                -1,
                2
            )

            points[:, 0] *= width
            points[:, 1] *= height

            points = np.round(
                points
            ).astype(np.int32)

            cv2.fillPoly(
                mask,
                [points],
                1
            )

    return mask


# ============================================================
# MASK → CROPPED IMAGE
# ============================================================

def crop_from_mask(
    image,
    mask,
    margin=0.20
):
    """
    Crop the original image around the segmentation mask.

    A margin is added around the bounding box to retain
    surrounding thyroid tissue.
    """

    if mask is None:
        return None

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    pad_x = int(
        width * margin
    )

    pad_y = int(
        height * margin
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        image.shape[1] - 1,
        x2 + pad_x
    )

    y2 = min(
        image.shape[0] - 1,
        y2 + pad_y
    )

    crop = image[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    return crop


# ============================================================
# DATASET
# ============================================================

class SegmentationGuidedDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform,
        image_root,
        label_root
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform

        self.image_root = Path(
            image_root
        )

        self.label_root = Path(
            label_root
        )

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        split = row["split"]

        image_path = (
            self.image_root
            / split
            / row["image_name"]
        )

        label_path = (
            self.label_root
            / split
            / f"{Path(row['image_name']).stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        height, width = image.shape[:2]

        mask = yolo_label_to_mask(
            label_path,
            width,
            height
        )

        crop = crop_from_mask(
            image,
            mask,
            CROP_MARGIN
        )

        # This should never happen for the clean dataset.
        # Fall back to the full image rather than crashing.
        if crop is None:
            crop = image

        crop = Image.fromarray(
            crop
        )

        if self.transform:
            crop = self.transform(
                crop
            )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        # Benign = 0
        # Malignant = 1
        benign_malignant = int(
            row["benign_malignant"]
        )

        # Non-PTC = 0
        # PTC = 1
        #
        # Missing histopathology = -1
        ptc = row["ptc"]

        if pd.isna(ptc):
            ptc = -1
        else:
            ptc = int(ptc)

        # FNAC:
        # Original classes 1-6
        # Convert to 0-5
        #
        # Missing = -1
        fnac = row["fnac_class"]

        if pd.isna(fnac):
            fnac = -1
        else:
            fnac = int(fnac) - 1

        # TIRADS:
        # Original classes 1-5
        # Convert to 0-4
        tirads = (
            int(row["tirads_class"])
            - 1
        )

        targets = {
            "benign_malignant":
                benign_malignant,

            "ptc":
                ptc,

            "fnac":
                fnac,

            "tirads":
                tirads,
        }

        return (
            crop,
            targets
        )


# ============================================================
# EFFICIENTNET-B0 MULTI-TASK MODEL
# ============================================================

class MultiTaskEfficientNet(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        try:

            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
            )

            backbone = (
                models.efficientnet_b0(
                    weights=weights
                )
            )

            print(
                "Loaded ImageNet-pretrained "
                "EfficientNet-B0."
            )

        except Exception as e:

            print(
                "\nWARNING: Could not load "
                "ImageNet pretrained weights."
            )

            print(
                f"Reason: {e}"
            )

            print(
                "Falling back to randomly "
                "initialized EfficientNet-B0."
            )

            backbone = (
                models.efficientnet_b0(
                    weights=None
                )
            )

        num_features = (
            backbone.classifier[1].in_features
        )

        backbone.classifier = nn.Identity()

        self.backbone = backbone

        self.head_benign_malignant = (
            nn.Linear(
                num_features,
                2
            )
        )

        self.head_ptc = (
            nn.Linear(
                num_features,
                2
            )
        )

        self.head_fnac = (
            nn.Linear(
                num_features,
                6
            )
        )

        self.head_tirads = (
            nn.Linear(
                num_features,
                5
            )
        )

    def forward(
        self,
        x
    ):

        features = self.backbone(
            x
        )

        return {
            "benign_malignant":
                self.head_benign_malignant(
                    features
                ),

            "ptc":
                self.head_ptc(
                    features
                ),

            "fnac":
                self.head_fnac(
                    features
                ),

            "tirads":
                self.head_tirads(
                    features
                ),
        }


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(
    df,
    column,
    num_classes,
    offset=0
):
    """
    Calculate inverse-frequency class weights.

    Missing labels are excluded.
    """

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    values = values.dropna()

    values = values.astype(int)

    if offset:
        values = values - offset

    counts = np.bincount(
        values,
        minlength=num_classes
    )

    counts = np.maximum(
        counts,
        1
    )

    weights = len(values) / (
        num_classes * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )


# ============================================================
# TRAINING LOSS
# ============================================================
class MultiTaskLoss:
    def __init__(self, train_df):
        self.bm_weight = get_class_weights(
            train_df,
            "benign_malignant",
            2
        )

        self.ptc_weight = get_class_weights(
            train_df[
                train_df["ptc"].notna()
            ],
            "ptc",
            2
        )

        self.fnac_weight = get_class_weights(
            train_df[
                train_df["fnac_class"].notna()
            ],
            "fnac_class",
            6,
            offset=1
        )

        self.tirads_weight = get_class_weights(
            train_df,
            "tirads_class",
            5,
            offset=1
        )

    def safe_cross_entropy(
        self,
        logits,
        targets,
        weight
    ):
        valid = targets >= 0

        if not valid.any():
            return torch.zeros(
                (),
                device=logits.device,
                dtype=logits.dtype
            )

        return nn.functional.cross_entropy(
            logits[valid],
            targets[valid],
            weight=weight,
            reduction="mean"
        )

    def __call__(
        self,
        outputs,
        targets
    ):
        bm_target = targets[
            "benign_malignant"
        ].to(DEVICE)

        ptc_target = targets[
            "ptc"
        ].to(DEVICE)

        fnac_target = targets[
            "fnac"
        ].to(DEVICE)

        tirads_target = targets[
            "tirads"
        ].to(DEVICE)

        bm_loss = self.safe_cross_entropy(
            outputs["benign_malignant"],
            bm_target,
            self.bm_weight
        )

        ptc_loss = self.safe_cross_entropy(
            outputs["ptc"],
            ptc_target,
            self.ptc_weight
        )

        fnac_loss = self.safe_cross_entropy(
            outputs["fnac"],
            fnac_target,
            self.fnac_weight
        )

        tirads_loss = self.safe_cross_entropy(
            outputs["tirads"],
            tirads_target,
            self.tirads_weight
        )

        total = (
            bm_loss
            + ptc_loss
            + fnac_loss
            + tirads_loss
        )

        return total


# ============================================================
# EPOCH
# ============================================================
def run_epoch(
    model,
    loader,
    loss_function,
    optimizer=None
):

    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0

    predictions = {
        task: []
        for task in [
            "benign_malignant",
            "ptc",
            "fnac",
            "tirads"
        ]
    }

    targets_all = {
        task: []
        for task in predictions
    }

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):

            outputs = model(
                images
            )

            loss = loss_function(
                outputs,
                targets
            )

            if not torch.isfinite(loss):
                print("\nWarning: Non-finite loss detected.")
                print(f"Loss: {loss.item()}")
                continue

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        total_loss += (
            loss.item()
        )

        batches += 1

        for task in predictions:

            pred = (
                outputs[task]
                .argmax(
                    dim=1
                )
                .detach()
                .cpu()
                .numpy()
            )

            true = (
                targets[task]
                .numpy()
            )

            predictions[task].extend(
                pred
            )

            targets_all[task].extend(
                true
            )

    metrics = {}

    for task in predictions:

        y_pred = np.asarray(
            predictions[task]
        )

        y_true = np.asarray(
            targets_all[task]
        )

        # Missing labels = -1.
        valid = (
            y_true >= 0
        )

        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:

            metrics[task] = {
                "macro_f1": np.nan,
                "accuracy": np.nan,
                "n": 0
            }

            continue

        metrics[task] = {
            "macro_f1": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "accuracy": accuracy_score(
                y_true,
                y_pred
            ),

            "n": len(y_true)
        }

    return (
        total_loss / max(
            batches,
            1
        ),
        metrics
    )


# ============================================================
# TEST SET WITH YOLO PREDICTED MASKS
# ============================================================

def create_test_crops(
    df,
    seg_model
):

    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("GENERATING TEST CROPS FROM YOLO PREDICTED MASKS")
    print("=" * 70)

    test_crop_dir = (
        OUTPUT
        / "test_crops"
    )

    test_crop_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model = YOLO(
        str(seg_model)
    )

    rows = []

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        image_path = (
            IMAGE_ROOT
            / "test"
            / row["image_name"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: failed to read "
                f"{image_path.name}"
            )

            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONF,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )

        result = results[0]

        pred_mask = np.zeros(
            image_rgb.shape[:2],
            dtype=np.uint8
        )

        if result.masks is not None:

            for polygon in result.masks.xy:

                polygon = np.asarray(
                    polygon,
                    dtype=np.float32
                )

                if len(polygon) < 3:
                    continue

                polygon = np.round(
                    polygon
                ).astype(np.int32)

                cv2.fillPoly(
                    pred_mask,
                    [polygon],
                    1
                )

        crop = crop_from_mask(
            image_rgb,
            pred_mask,
            CROP_MARGIN
        )

        # If YOLO fails to produce a mask, use the
        # complete image. This preserves the sample
        # for classification and records the failure.
        segmentation_failed = (
            crop is None
        )

        if crop is None:
            crop = image_rgb

        crop_path = (
            test_crop_dir
            / row["image_name"]
        )

        Image.fromarray(
            crop
        ).save(
            crop_path
        )

        new_row = row.to_dict()

        new_row[
            "crop_path"
        ] = str(crop_path)

        new_row[
            "segmentation_failed"
        ] = int(
            segmentation_failed
        )

        rows.append(
            new_row
        )

        if i % 100 == 0:
            print(
                f"Processed "
                f"{i}/{len(df)} test images"
            )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        OUTPUT
        / "test_classification_metadata.csv",
        index=False
    )

    print(
        f"\nTest crops saved to:\n"
        f"{test_crop_dir}"
    )

    print(
        f"YOLO segmentation failures: "
        f"{result_df['segmentation_failed'].sum()}"
    )

    return result_df


# ============================================================
# TEST DATASET FROM PREGENERATED CROPS
# ============================================================

class TestCropDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform

    def __len__(
        self
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image = Image.open(
            row["crop_path"]
        ).convert(
            "RGB"
        )

        image = self.transform(
            image
        )

        targets = {
            "benign_malignant":
                int(
                    row[
                        "benign_malignant"
                    ]
                ),

            "ptc":
                -1
                if pd.isna(
                    row["ptc"]
                )
                else int(
                    row["ptc"]
                ),

            "fnac":
                -1
                if pd.isna(
                    row["fnac_class"]
                )
                else int(
                    row["fnac_class"]
                ) - 1,

            "tirads":
                int(
                    row["tirads_class"]
                ) - 1,
        }

        return (
            image,
            targets
        )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_test(
    model,
    test_df,
    loss_function
):

    dataset = TestCropDataset(
        test_df,
        eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loss, metrics = run_epoch(
        model,
        loader,
        loss_function
    )

    print("\n" + "=" * 70)
    print("OFFICIAL TEST CLASSIFICATION RESULTS")
    print("=" * 70)

    results = []

    for task in [
        "benign_malignant",
        "ptc",
        "fnac",
        "tirads"
    ]:

        result = {
            "task": task,
            "macro_f1":
                float(
                    metrics[task][
                        "macro_f1"
                    ]
                ),
            "accuracy":
                float(
                    metrics[task][
                        "accuracy"
                    ]
                ),
            "n":
                int(
                    metrics[task]["n"]
                )
        }

        results.append(
            result
        )

        print(
            f"\n{task}:"
        )

        print(
            f"  Macro F1 = "
            f"{result['macro_f1']:.4f}"
        )

        print(
            f"  Accuracy = "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"  N = "
            f"{result['n']}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT
        / "official_test_classification_results.csv",
        index=False
    )

    with open(
        OUTPUT
        / "official_test_classification_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEGMENTATION-GUIDED MULTI-TASK CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not METADATA.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA}"
        )

    if not SEG_MODEL.exists():

        raise FileNotFoundError(
            f"YOLO best.pt not found:\n{SEG_MODEL}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"\nTrain: {len(train_df)}"
    )

    print(
        f"Val:   {len(val_df)}"
    )

    print(
        f"Test:  {len(test_df)}"
    )

    # --------------------------------------------------------
    # Training datasets
    # --------------------------------------------------------

    train_dataset = (
        SegmentationGuidedDataset(
            train_df,
            train_transform,
            IMAGE_ROOT,
            LABEL_ROOT
        )
    )

    val_dataset = (
        SegmentationGuidedDataset(
            val_df,
            eval_transform,
            IMAGE_ROOT,
            LABEL_ROOT
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        MultiTaskEfficientNet()
        .to(DEVICE)
    )

    loss_function = MultiTaskLoss(
        train_df
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    history = []

    best_val_loss = float(
        "inf"
    )

    best_state = None

    no_improvement = 0

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        train_loss, train_metrics = (
            run_epoch(
                model,
                train_loader,
                loss_function,
                optimizer
            )
        )

        val_loss, val_metrics = (
            run_epoch(
                model,
                val_loader,
                loss_function
            )
        )

        scheduler.step(
            val_loss
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr":
                optimizer.param_groups[
                    0
                ]["lr"]
        }

        for task in [
            "benign_malignant",
            "ptc",
            "fnac",
            "tirads"
        ]:

            row[
                f"val_{task}_macro_f1"
            ] = val_metrics[
                task
            ]["macro_f1"]

        history.append(
            row
        )

        print(
            f"\nEpoch "
            f"{epoch}/{EPOCHS}"
        )

        print(
            f"  train loss = "
            f"{train_loss:.4f}"
        )

        print(
            f"  val loss   = "
            f"{val_loss:.4f}"
        )

        print(
            f"  time       = "
            f"{elapsed:.1f}s"
        )

        print(
            "  validation Macro F1:"
        )

        for task in [
            "benign_malignant",
            "ptc",
            "fnac",
            "tirads"
        ]:

            print(
                f"    {task:20s} "
                f"{val_metrics[task]['macro_f1']:.4f}"
            )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict":
                        best_state,

                    "epoch":
                        epoch,

                    "val_loss":
                        best_val_loss
                },
                OUTPUT
                / "best_classifier.pt"
            )

            no_improvement = 0

            print(
                "  -> best classifier saved"
            )

        else:

            no_improvement += 1

        if no_improvement >= PATIENCE:

            print(
                "\nEarly stopping."
            )

            break

    total_time = (
        time.time()
        - start_time
    )

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT
        / "training_history.csv",
        index=False
    )

    print(
        f"\nClassifier training time: "
        f"{total_time / 3600:.2f} h"
    )

    # --------------------------------------------------------
    # Load best classifier
    # --------------------------------------------------------

    checkpoint = torch.load(
        OUTPUT
        / "best_classifier.pt",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"\nLoaded best classifier "
        f"from epoch "
        f"{checkpoint['epoch']}"
    )

    # --------------------------------------------------------
    # Generate official test crops
    # USING YOLO PREDICTED SEGMENTATION
    # --------------------------------------------------------

    test_df_with_crops = (
        create_test_crops(
            test_df,
            SEG_MODEL
        )
    )

    # --------------------------------------------------------
    # Official test evaluation
    # --------------------------------------------------------

    evaluate_test(
        model,
        test_df_with_crops,
        loss_function
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput directory:\n{OUTPUT}"
    )


if __name__ == "__main__":
    main()