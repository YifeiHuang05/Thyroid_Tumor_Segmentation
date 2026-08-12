from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26n-seg training, optional HPO, and evaluation for ThyroidXL."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset.yaml"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to local yolo26n-seg.pt"
    )

    parser.add_argument(
        "--project",
        default=None,
        help="Output project directory"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Final training epochs"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. -1 lets Ultralytics auto-select."
    )

    parser.add_argument(
        "--device",
        default="0",
        help="GPU device, e.g. 0 or 0,1"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader workers"
    )

    parser.add_argument(
        "--run-tune",
        action="store_true",
        help="Run Ultralytics model.tune() before final training."
    )

    parser.add_argument(
        "--tune-iterations",
        type=int,
        default=10,
        help="Number of HPO iterations."
    )

    parser.add_argument(
        "--tune-epochs",
        type=int,
        default=20,
        help="Epochs per HPO trial."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for image-level PTC classification."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate final model on test set."
    )

    return parser.parse_args()


# ============================================================
# CUDA CHECK
# ============================================================

def check_cuda(device):

    print("=" * 70)
    print("SYSTEM CHECK")
    print("=" * 70)

    print(f"PyTorch version:       {torch.__version__}")
    print(f"PyTorch CUDA version:  {torch.version.cuda}")
    print(f"CUDA available:        {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Do not start training until PyTorch can see the GPU."
        )

    print(
        f"GPU:                   "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Requested device:      {device}"
    )

    print("=" * 70)


# ============================================================
# TRAINING
# ============================================================
def train_model(
    model_path,
    dataset,
    project,
    epochs,
    batch,
    device,
    workers,
    name,
    hyperparameters=None,
):

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model = YOLO(
        str(model_path)
    )

    train_args = {
        "data": str(dataset),
        "epochs": epochs,

        # Keep default YOLO26 image size.
        # imgsz deliberately omitted.

        "batch": batch,
        "device": device,
        "workers": workers,

        "project": str(project),
        "name": name,
        "exist_ok": True,

        "pretrained": True,

        "plots": True,
        "save": True,
        "val": True,
    }

    if hyperparameters:
        train_args.update(
            hyperparameters
        )

    print("\nTraining arguments:")

    for key, value in train_args.items():
        print(
            f"  {key}: {value}"
        )

    results = model.train(
        **train_args
    )

    return model, results


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def run_tuning(
    model_path,
    dataset,
    project,
    iterations,
    epochs,
    batch,
    device,
    workers,
):

    print("\n" + "=" * 70)
    print("YOLO26n-seg HYPERPARAMETER TUNING")
    print("=" * 70)

    print(
        f"Iterations: {iterations}"
    )

    print(
        f"Epochs/trial: {epochs}"
    )

    print(
        "NOTE: this uses the validation set for HPO."
    )

    model = YOLO(
        str(model_path)
    )

    tune_dir = (
        Path(project) /
        "tune"
    )

    model.tune(
        data=str(dataset),
        epochs=epochs,
        iterations=iterations,

        batch=batch,
        device=device,
        workers=workers,

        project=str(project),
        name="tune",

        plots=True,
        save=True,
        val=True,
    )

    best_yaml = (
        tune_dir /
        "best_hyperparameters.yaml"
    )

    if not best_yaml.exists():

        raise FileNotFoundError(
            f"Could not find:\n{best_yaml}\n"
            "Check the tuning output directory."
        )

    print(
        f"\nBest hyperparameters:\n"
        f"{best_yaml}"
    )

    return best_yaml


# ============================================================
# LOAD YAML HYPERPARAMETERS
# ============================================================
def load_yaml(path):

    import yaml

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# ULTRALYTICS STANDARD VALIDATION
# ============================================================
def standard_validation(
    model_path,
    dataset,
    device,
    project,
):

    print("\n" + "=" * 70)
    print("ULTRALYTICS STANDARD TEST EVALUATION")
    print("=" * 70)

    model = YOLO(
        str(model_path)
    )

    metrics = model.val(
        data=str(dataset),
        split="test",
        device=device,
        plots=True,
        project=str(project),
        name="test_metrics",
        exist_ok=True,
    )

    print("\nStandard Ultralytics metrics:")

    try:
        print(
            f"Box mAP50:        "
            f"{metrics.box.map50:.4f}"
        )

        print(
            f"Box mAP50-95:     "
            f"{metrics.box.map:.4f}"
        )
    except Exception:
        pass

    try:
        print(
            f"Mask mAP50:       "
            f"{metrics.seg.map50:.4f}"
        )

        print(
            f"Mask mAP50-95:    "
            f"{metrics.seg.map:.4f}"
        )
    except Exception:
        pass

    return metrics


# ============================================================
# IMAGE-LEVEL PTC CLASSIFICATION
# ============================================================
def load_ground_truth_labels(
    dataset_root,
):
    """
    Determine image-level PTC ground truth from YOLO labels.

    Positive:
        label file contains >=1 object.

    Negative:
        label file is empty.

    Returns:
        dict[image_stem] -> 0/1
    """
    image_dir = (
        Path(dataset_root) /
        "images" /
        "test"
    )

    label_dir = (
        Path(dataset_root) /
        "labels" /
        "test"
    )

    ground_truth = {}

    for image_path in sorted(
        image_dir.iterdir()
    ):

        if not image_path.is_file():
            continue

        label_path = (
            label_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label file:\n{label_path}"
            )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        ground_truth[
            image_path.stem
        ] = int(
            len(lines) > 0
        )

    return ground_truth


def image_level_evaluation(
    model_path,
    dataset_root,
    device,
    conf,
    project,
):

    print("\n" + "=" * 70)
    print("IMAGE-LEVEL PTC CLASSIFICATION")
    print("=" * 70)

    model = YOLO(
        str(model_path)
    )

    ground_truth = load_ground_truth_labels(
        dataset_root
    )

    image_dir = (
        Path(dataset_root) /
        "images" /
        "test"
    )

    y_true = []
    y_pred = []

    inference_times = []

    output_rows = []

    for image_path in sorted(
        image_dir.iterdir()
    ):

        if not image_path.is_file():
            continue

        stem = image_path.stem

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            conf=conf,
            device=device,
            verbose=False,
            save=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        inference_ms = (
            elapsed * 1000
        )

        inference_times.append(
            inference_ms
        )

        # ----------------------------------------------------
        # Determine image-level prediction
        # ----------------------------------------------------

        result = results[0]

        predicted_positive = False
        n_predictions = 0

        if result.boxes is not None:

            n_predictions = len(
                result.boxes
            )

            if n_predictions > 0:
                predicted_positive = True

        gt = ground_truth[stem]

        pred = int(
            predicted_positive
        )

        y_true.append(gt)
        y_pred.append(pred)

        output_rows.append({
            "image": image_path.name,
            "ground_truth": gt,
            "prediction": pred,
            "n_predictions": n_predictions,
            "inference_ms": inference_ms,
        })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    mean_inference_ms = np.mean(
        inference_times
    )

    median_inference_ms = np.median(
        inference_times
    )

    p95_inference_ms = np.percentile(
        inference_times,
        95,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nImage-level metrics:")
    print(
        f"Accuracy:             {accuracy:.4f}"
    )
    print(
        f"Balanced accuracy:    {balanced_accuracy:.4f}"
    )
    print(
        f"Macro F1:             {macro_f1:.4f}"
    )
    print(
        f"F1:                   {f1:.4f}"
    )
    print(
        f"Precision:            {precision:.4f}"
    )
    print(
        f"Sensitivity/Recall:   {sensitivity:.4f}"
    )
    print(
        f"Specificity:          {specificity:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        "                 Predicted"
    )
    print(
        "                 Negative Positive"
    )
    print(
        f"Actual Negative  {tn:8d} {fp:8d}"
    )
    print(
        f"Actual Positive  {fn:8d} {tp:8d}"
    )

    print("\nInference time:")
    print(
        f"Mean:               {mean_inference_ms:.2f} ms/image"
    )
    print(
        f"Median:             {median_inference_ms:.2f} ms/image"
    )
    print(
        f"P95:                {p95_inference_ms:.2f} ms/image"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_dir = (
        Path(project) /
        "test_metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir /
        "image_level_predictions.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ground_truth",
                "prediction",
                "n_predictions",
                "inference_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    metrics_path = (
        output_dir /
        "image_level_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "metric",
            "value",
        ])

        writer.writerow([
            "accuracy",
            accuracy,
        ])

        writer.writerow([
            "balanced_accuracy",
            balanced_accuracy,
        ])

        writer.writerow([
            "macro_f1",
            macro_f1,
        ])

        writer.writerow([
            "f1",
            f1,
        ])

        writer.writerow([
            "precision",
            precision,
        ])

        writer.writerow([
            "sensitivity",
            sensitivity,
        ])

        writer.writerow([
            "specificity",
            specificity,
        ])

        writer.writerow([
            "tn",
            tn,
        ])

        writer.writerow([
            "fp",
            fp,
        ])

        writer.writerow([
            "fn",
            fn,
        ])

        writer.writerow([
            "tp",
            tp,
        ])

        writer.writerow([
            "mean_inference_ms",
            mean_inference_ms,
        ])

        writer.writerow([
            "median_inference_ms",
            median_inference_ms,
        ])

        writer.writerow([
            "p95_inference_ms",
            p95_inference_ms,
        ])

    # --------------------------------------------------------
    # Save confusion matrix as simple CSV
    # --------------------------------------------------------

    cm_path = (
        output_dir /
        "confusion_matrix.csv"
    )

    np.savetxt(
        cm_path,
        cm,
        delimiter=",",
        fmt="%d",
    )

    print("\nSaved:")
    print(
        f"  {csv_path}"
    )
    print(
        f"  {metrics_path}"
    )
    print(
        f"  {cm_path}"
    )

    # Save confusion matrix as figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(cm_path.with_suffix(".png"))
    plt.close()
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mean_inference_ms": mean_inference_ms,
        "median_inference_ms": median_inference_ms,
        "p95_inference_ms": p95_inference_ms,
        "confusion_matrix": cm,
    }

    

# ============================================================
# MAIN
# ============================================================
def main():

    args = parse_args()

    dataset = Path(
        args.dataset
    ).resolve()

    model_path = Path(
        args.model
    ).resolve()

    if not dataset.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found:\n{dataset}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    dataset_root = Path(
        dataset.parent
    )

    if args.project:
        project = Path(
            args.project
        ).resolve()
    else:
        project = (
            dataset_root /
            "runs"
        )

    project.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_cuda(
        args.device
    )

    # --------------------------------------------------------
    # Optional HPO
    # --------------------------------------------------------

    best_hyperparameters = None

    if args.run_tune:

        best_yaml = run_tuning(
            model_path=model_path,
            dataset=dataset,
            project=project,
            iterations=args.tune_iterations,
            epochs=args.tune_epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )

        best_hyperparameters = load_yaml(
            best_yaml
        )

        print(
            "\nUsing tuned hyperparameters:"
        )

        for key, value in (
            best_hyperparameters.items()
        ):
            print(
                f"  {key}: {value}"
            )

    # --------------------------------------------------------
    # Final training
    # --------------------------------------------------------

    final_model, results = train_model(
        model_path=model_path,
        dataset=dataset,
        project=project,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name="final",
        hyperparameters=best_hyperparameters,
    )

    best_model_path = (
        project /
        "final" /
        "weights" /
        "best.pt"
    )

    print(
        f"\nBest model:\n"
        f"{best_model_path}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    if args.test:

        standard_validation(
            model_path=best_model_path,
            dataset=dataset,
            device=args.device,
            project=project,
        )

        image_level_evaluation(
            model_path=best_model_path,
            dataset_root=dataset_root,
            device=args.device,
            conf=args.conf,
            project=project,
        )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()