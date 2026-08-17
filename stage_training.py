from pathlib import Path
import shutil

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent


def _build_stage2_classification_dataset() -> Path:
    src_root = PROJECT_ROOT / "data" / "ThyroidXL_two_stage" / "stage2"
    out_root = PROJECT_ROOT / "data" / "ThyroidXL_two_stage" / "stage2_cls"
    class_map = {"0": "PTC", "1": "NonPTC"}

    for split in ("train", "val", "test"):
        for class_name in class_map.values():
            (out_root / split / class_name).mkdir(parents=True, exist_ok=True)

        labels_dir = src_root / "labels" / split
        images_dir = src_root / "images" / split

        for label_path in sorted(labels_dir.glob("*.txt")):
            label_value = label_path.read_text(encoding="utf-8").strip()
            class_name = class_map.get(label_value)
            if class_name is None:
                continue

            stem = label_path.stem
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                for candidate in images_dir.glob(f"{stem}.*"):
                    if candidate.is_file():
                        image_path = candidate
                        break

            if image_path is None:
                continue

            target_path = out_root / split / class_name / image_path.name
            shutil.copy2(image_path, target_path)

    return out_root


def train_stage1():
    model = YOLO(str(PROJECT_ROOT / "model" / "yolo26n-seg.pt"))
    results = model.train(
        data=str(PROJECT_ROOT / "data" / "ThyroidXL_two_stage" / "stage1" / "stage1.yaml"),
        epochs=50,
        batch=8,
        #imgsz=640,
        device=0,
        project=str(PROJECT_ROOT / "runs_stage1"),
        name="stage1_final",
        exist_ok=True,
        pretrained=True,
        val=True,
        plots=True,
        verbose=True,
    )
    return results


def train_stage2():
    stage2_cls_root = _build_stage2_classification_dataset()
    model = YOLO(str(PROJECT_ROOT / "model" / "yolov8n-cls.pt"))
    results = model.train(
        data=str(stage2_cls_root),
        epochs=50,
        batch=16,
        #imgsz=224,
        device=0,
        project=str(PROJECT_ROOT / "runs_stage2"),
        name="stage2_final",
        exist_ok=True,
        pretrained=True,
        val=True,
        plots=True,
        verbose=True,
    )
    return results


if __name__ == "__main__":
    train_stage1()
    train_stage2()
