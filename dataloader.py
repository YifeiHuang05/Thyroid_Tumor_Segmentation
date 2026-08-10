"""
dataloader.py

ThyroidXL Classification DataLoader
Final QC Candidate - 448x448 version
============================================================

COCO JSON structure:

categories:
    0 -> benign
    1 -> malignant

images:
    {
        "id": 1,
        "file_name": "...png",
        "height": 532,
        "width": 727,
        "patient_id": 58,
        ...
    }

annotations:
    {
        "id": 1,
        "image_id": 1,
        "category_id": 0,
        "bbox": [...],
        "segmentation": [...],
        ...
    }

Classification mapping:

images.id
    ↓
annotations.image_id
    ↓
annotations.category_id
    ↓
0 = benign
1 = malignant


Directory:

C:\\ThyroidXL
│
├── dataloader.py
│
├── check_augmentation.py
│
└── ThyroidXL
    ├── train
    │   ├── train_annotations.json
    │   └── images
    └── test
        ├── test_annotations.json
        └── images


Final preprocessing:

Original image
    ↓
PadToSquare (preserve aspect ratio)
    ↓
Resize 448 × 448
    ↓
very conservative augmentation (train only)
    ↓
ToTensor
    ↓
ImageNet Normalize


Default:
    image_size = 448
    batch_size = 4

Labels:
    0 = benign
    1 = malignant
"""

# =========
# Imports
# =========

import json
import random
import warnings

from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict, Union

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from PIL import (
    Image,
    ImageFile,
    ImageOps,
)

from torchvision import transforms

from torchvision.transforms import InterpolationMode


# ========
# PIL
# ========

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========
# Paths
# ==========

PROJECT_ROOT = Path(__file__).resolve().parent

# ================= 路径调整 =================
DATA_ROOT = PROJECT_ROOT / "ThyroidXL"

TRAIN_DIR = DATA_ROOT / "train"
TRAIN_JSON = TRAIN_DIR / "train_annotations.json"
TRAIN_IMAGE_DIR = TRAIN_DIR / "images"

TEST_DIR = DATA_ROOT / "test"
TEST_JSON = TEST_DIR / "test_annotations.json"
TEST_IMAGE_DIR = TEST_DIR / "images"


# =============
# Defaults
# =============

DEFAULT_IMAGE_SIZE = 448

DEFAULT_BATCH_SIZE = 4

# Windows first test: 0
DEFAULT_NUM_WORKERS = 0

DEFAULT_PIN_MEMORY = True


# =========
# Labels
# =========

LABEL_TO_NAME = {
    0: "benign",
    1: "malignant",
}

NAME_TO_LABEL = {
    "benign": 0,
    "malignant": 1,
}


# ==============================
# ImageNet normalization
# ==============================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# =============================
# Pad To Square
# =============================

class PadToSquare:
    """
    Preserve aspect ratio by padding the shorter side.

    Example:

        727 x 532
            ↓
        727 x 727
            ↓
        Resize 448 x 448

    fill=0 uses black padding.

    For ultrasound images, black padding is preferred over reflection padding 
    because reflection could create artificial anatomical structures.
    """

    def __init__(
        self,
        fill: int = 0,
    ):
        self.fill = fill

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:

        width, height = image.size

        if width == height:
            return image

        if width > height:

            difference = width - height

            top = difference // 2
            bottom = difference - top

            left = 0
            right = 0

        else:

            difference = height - width

            left = difference // 2
            right = difference - left

            top = 0
            bottom = 0

        return ImageOps.expand(
            image,
            border=(
                left,
                top,
                right,
                bottom,
            ),
            fill=self.fill,
        )


# ============================================================
# Speckle noise
# ============================================================

class RandomSpeckleNoise:
    """
    Very mild multiplicative speckle noise.

    Ultrasound images already contain natural speckle,
    therefore this augmentation is intentionally weak.

    noisy = image + image * noise
    """

    def __init__(
        self,
        probability: float = 0.10,
        std_range: Tuple[float, float] = (
            0.005,
            0.015,
        ),
    ):

        self.probability = probability
        self.std_range = std_range

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        if random.random() > self.probability:
            return image

        std = random.uniform(
            self.std_range[0],
            self.std_range[1],
        )

        noise = (
            torch.randn_like(image)
            * std
        )

        noisy = (
            image
            + image * noise
        )

        return torch.clamp(
            noisy,
            0.0,
            1.0,
        )


# ============================================================
# Base preprocessing
# ============================================================

def build_preprocess_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    """
    No random augmentation.

    Used by:
    - test / evaluation
    - QC preprocessing comparison

    Output is normalized tensor.
    """

    return transforms.Compose([

        PadToSquare(
            fill=0,
        ),

        transforms.Resize(
            (
                image_size,
                image_size,
            ),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# Display preprocessing
# ============================================================

def build_display_preprocess_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    """
    Same spatial preprocessing as the model,
    but WITHOUT normalization.

    Intended only for visualization / QC.
    """

    return transforms.Compose([

        PadToSquare(
            fill=0,
        ),

        transforms.Resize(
            (
                image_size,
                image_size,
            ),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ])


# ============================================================
# Train transform
# ============================================================

def build_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    """
    Conservative thyroid ultrasound augmentation.

    Compared with previous versions:

    Contrast:
        ±10%, p 0.30

    Gaussian blur:
        sigma 0.1–0.3, p 0.05

    Elastic:
        alpha 3, p 0.03

    Speckle:
        std 0.005–0.015, p 0.10

    Most images therefore stay very close to the source image.
    """

    return transforms.Compose([

        # ----------------------------------------------------
        # 1. Preserve aspect ratio
        # ----------------------------------------------------

        PadToSquare(
            fill=0,
        ),

        # ----------------------------------------------------
        # 2. Resize
        # ----------------------------------------------------

        transforms.Resize(
            (
                image_size,
                image_size,
            ),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),

        # ----------------------------------------------------
        # 3. Mild contrast
        # ----------------------------------------------------

        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    contrast=(
                        0.90,
                        1.10,
                    ),
                )
            ],
            p=0.30,
        ),

        # ----------------------------------------------------
        # 4. Extremely mild Gaussian blur
        # ----------------------------------------------------

        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(
                        0.10,
                        0.30,
                    ),
                )
            ],
            p=0.05,
        ),

        # ----------------------------------------------------
        # 5. Extremely mild elastic deformation
        # ----------------------------------------------------

        transforms.RandomApply(
            [
                transforms.ElasticTransform(
                    alpha=3.0,
                    sigma=4.0,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            ],
            p=0.03,
        ),

        # ----------------------------------------------------
        # 6. PIL -> Tensor
        # ----------------------------------------------------

        transforms.ToTensor(),

        # ----------------------------------------------------
        # 7. Very mild speckle noise
        # ----------------------------------------------------

        RandomSpeckleNoise(
            probability=0.10,
            std_range=(
                0.005,
                0.015,
            ),
        ),

        # ----------------------------------------------------
        # 8. Normalize
        # ----------------------------------------------------

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# Eval transform
# ============================================================

def build_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):

    return build_preprocess_transform(
        image_size=image_size
    )


# ============================================================
# Load JSON
# ============================================================

def load_json(
    json_path: Union[str, Path],
) -> dict:

    json_path = Path(
        json_path
    )

    if not json_path.exists():

        raise FileNotFoundError(
            "\nJSON 文件不存在：\n"
            f"{json_path}\n"
        )

    print()
    print("正在读取 JSON：")
    print(json_path)

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    return data


# ============================================================
# Resolve image path
# ============================================================

def resolve_image_path(
    file_name: str,
    image_dir: Path,
    split_dir: Path,
) -> Optional[Path]:

    candidates = [

        image_dir / file_name,

        split_dir / file_name,

        image_dir / Path(file_name).name,

        split_dir / Path(file_name).name,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return None


# ============================================================
# Parse COCO classification data
# ============================================================

def parse_coco_classification(
    json_path: Union[str, Path],
    image_dir: Union[str, Path],
    require_label: bool = True,
    verify_images: bool = True,
) -> Tuple[
    List[Dict],
    Dict[int, str],
]:

    json_path = Path(
        json_path
    )

    image_dir = Path(
        image_dir
    )

    split_dir = json_path.parent

    data = load_json(
        json_path
    )

    if "images" not in data:

        raise RuntimeError(
            "\nJSON 中不存在 images 字段。\n"
            f"顶层字段：{list(data.keys())}"
        )

    images = data.get(
        "images",
        [],
    )

    annotations = data.get(
        "annotations",
        [],
    )

    categories = data.get(
        "categories",
        [],
    )

    print()
    print("JSON 信息：")

    print(
        f"  images      : "
        f"{len(images)}"
    )

    print(
        f"  annotations : "
        f"{len(annotations)}"
    )

    print(
        f"  categories  : "
        f"{len(categories)}"
    )

    # ========================================================
    # Category map
    # ========================================================

    category_map = {}

    for category in categories:

        if not isinstance(
            category,
            dict,
        ):
            continue

        if "id" not in category:
            continue

        category_id = int(
            category["id"]
        )

        category_name = str(
            category.get(
                "name",
                f"class_{category_id}",
            )
        )

        category_map[
            category_id
        ] = category_name

    print()
    print("类别定义：")

    for category_id, category_name in sorted(
        category_map.items()
    ):

        print(
            f"  {category_id}"
            f" -> "
            f"{category_name}"
        )

    # ========================================================
    # image_id -> category ids
    # ========================================================

    annotation_by_image = defaultdict(
        list
    )

    for annotation in annotations:

        if not isinstance(
            annotation,
            dict,
        ):
            continue

        image_id = annotation.get(
            "image_id"
        )

        category_id = annotation.get(
            "category_id"
        )

        if image_id is None:
            continue

        if category_id is None:
            continue

        annotation_by_image[
            int(image_id)
        ].append(
            int(category_id)
        )

    # ========================================================
    # Samples
    # ========================================================

    samples = []

    missing_images = []

    no_label_images = []

    conflicting_labels = []

    for image_info in images:

        if not isinstance(
            image_info,
            dict,
        ):
            continue

        image_id = image_info.get(
            "id"
        )

        file_name = image_info.get(
            "file_name"
        )

        if image_id is None:
            continue

        if not file_name:
            continue

        image_id = int(
            image_id
        )

        category_ids = annotation_by_image.get(
            image_id,
            [],
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if len(category_ids) == 0:

            if require_label:

                no_label_images.append(
                    file_name
                )

                continue

            label = -1

            label_name = "unknown"

        else:

            label_counts = Counter(
                category_ids
            )

            label = (
                label_counts
                .most_common(1)[0][0]
            )

            if len(label_counts) > 1:

                conflicting_labels.append(
                    (
                        image_id,
                        file_name,
                        dict(label_counts),
                        label,
                    )
                )

            label_name = category_map.get(
                label,
                f"class_{label}",
            )

        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        image_path = resolve_image_path(
            file_name=file_name,
            image_dir=image_dir,
            split_dir=split_dir,
        )

        if image_path is None:

            missing_images.append(
                file_name
            )

            if verify_images:
                continue

            image_path = (
                image_dir
                / file_name
            )

        sample = {

            "image_id":
                image_id,

            "image_path":
                str(image_path),

            "file_name":
                file_name,

            "label":
                int(label),

            "label_name":
                label_name,

            "patient_id":
                image_info.get(
                    "patient_id"
                ),

            "height":
                image_info.get(
                    "height"
                ),

            "width":
                image_info.get(
                    "width"
                ),
        }

        samples.append(
            sample
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("数据解析完成")
    print("=" * 70)

    print(
        f"成功生成样本数："
        f"{len(samples)}"
    )

    if missing_images:

        print()
        print(
            f"找不到图片："
            f"{len(missing_images)}"
        )

        print("前 5 个：")

        for item in missing_images[:5]:
            print(
                f"  {item}"
            )

    if no_label_images:

        print()
        print(
            f"没有 annotation 标签："
            f"{len(no_label_images)}"
        )

    if conflicting_labels:

        warnings.warn(
            "\n存在图片对应多个不同 category_id。\n"
            "当前自动采用出现次数最多的类别。\n"
            f"冲突数量：{len(conflicting_labels)}"
        )

    if len(samples) == 0:

        raise RuntimeError(
            "\n没有成功生成任何样本。\n"
            f"JSON：{json_path}\n"
            f"Images：{image_dir}\n"
        )

    label_counter = Counter(
        sample["label"]
        for sample in samples
        if sample["label"] >= 0
    )

    print()
    print("类别数量：")

    for label, count in sorted(
        label_counter.items()
    ):

        label_name = category_map.get(
            label,
            str(label),
        )

        ratio = (
            count
            / len(samples)
            * 100
        )

        print(
            f"  {label} "
            f"({label_name:10s}) : "
            f"{count:6d} "
            f"({ratio:6.2f}%)"
        )

    print("=" * 70)

    return (
        samples,
        category_map,
    )


# ============================================================
# Dataset
# ============================================================

class ThyroidDataset(
    Dataset
):

    def __init__(
        self,
        json_path: Union[str, Path],
        image_dir: Union[str, Path],
        transform=None,
        require_label: bool = True,
        verify_images: bool = True,
        return_metadata: bool = False,
    ):

        super().__init__()

        self.json_path = Path(
            json_path
        )

        self.image_dir = Path(
            image_dir
        )

        self.transform = transform

        self.return_metadata = (
            return_metadata
        )

        (
            self.samples,
            self.category_map,
        ) = parse_coco_classification(
            json_path=self.json_path,
            image_dir=self.image_dir,
            require_label=require_label,
            verify_images=verify_images,
        )

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        image_path = sample[
            "image_path"
        ]

        label = sample[
            "label"
        ]

        try:

            with Image.open(
                image_path
            ) as image:

                image = image.convert(
                    "RGB"
                )

        except Exception as error:

            raise RuntimeError(
                "\n图片读取失败：\n"
                f"{image_path}\n"
                f"错误：{error}"
            ) from error

        if self.transform is not None:

            image = self.transform(
                image
            )

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        if self.return_metadata:

            metadata = {

                "image_id":
                    sample["image_id"],

                "file_name":
                    sample["file_name"],

                "image_path":
                    sample["image_path"],

                "label_name":
                    sample["label_name"],

                "patient_id":
                    sample["patient_id"],

                "original_height":
                    sample["height"],

                "original_width":
                    sample["width"],
            }

            return (
                image,
                label,
                metadata,
            )

        return (
            image,
            label,
        )


# ============================================================
# Create DataLoader
# ============================================================

def create_dataloader(
    json_path: Union[str, Path],
    image_dir: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    train: bool = True,
    shuffle: Optional[bool] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    require_label: bool = True,
    return_metadata: bool = False,
):

    if shuffle is None:
        shuffle = train

    if train:

        transform = build_train_transform(
            image_size=image_size
        )

    else:

        transform = build_eval_transform(
            image_size=image_size
        )

    dataset = ThyroidDataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=transform,
        require_label=require_label,
        verify_images=True,
        return_metadata=return_metadata,
    )

    actual_pin_memory = (
        pin_memory
        and torch.cuda.is_available()
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": actual_pin_memory,
        "drop_last": False,
    }

    # Only valid when num_workers > 0
    if num_workers > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    dataloader = DataLoader(
        **loader_kwargs
    )

    return dataloader


# ============================================================
# Build train + test
# ============================================================

def build_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    return_metadata: bool = False,
):

    print()
    print("#" * 70)
    print("创建 Train DataLoader")
    print("#" * 70)

    train_loader = create_dataloader(
        json_path=TRAIN_JSON,
        image_dir=TRAIN_IMAGE_DIR,
        batch_size=batch_size,
        image_size=image_size,
        train=True,
        shuffle=True,
        num_workers=num_workers,
        require_label=True,
        return_metadata=return_metadata,
    )

    print()
    print("#" * 70)
    print("创建 Test DataLoader")
    print("#" * 70)

    if TEST_JSON.exists():

        test_loader = create_dataloader(
            json_path=TEST_JSON,
            image_dir=TEST_IMAGE_DIR,
            batch_size=batch_size,
            image_size=image_size,
            train=False,
            shuffle=False,
            num_workers=num_workers,
            require_label=True,
            return_metadata=return_metadata,
        )

    else:

        print()
        print(
            "没有找到 Test JSON："
        )

        print(
            TEST_JSON
        )

        test_loader = None

    return (
        train_loader,
        test_loader,
    )


# ============================================================
# Environment
# ============================================================

def print_environment():

    print()
    print("=" * 70)
    print("PyTorch 环境")
    print("=" * 70)

    print(
        f"PyTorch        : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA Runtime   : "
        f"{torch.version.cuda}"
    )

    print(
        f"CUDA Available : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        memory_gb = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024 ** 3
        )

        print(
            f"GPU Memory      : "
            f"{memory_gb:.2f} GB"
        )

    else:

        print(
            "GPU             : CPU"
        )

    print("=" * 70)


# ============================================================
# Test one batch
# ============================================================

def test_one_batch(
    dataloader: DataLoader,
):

    print()
    print("=" * 70)
    print("测试 DataLoader")
    print("=" * 70)

    images, labels = next(
        iter(dataloader)
    )

    print(
        f"images.shape : "
        f"{images.shape}"
    )

    print(
        f"labels.shape : "
        f"{labels.shape}"
    )

    print(
        f"images dtype : "
        f"{images.dtype}"
    )

    print(
        f"labels dtype : "
        f"{labels.dtype}"
    )

    print(
        f"labels       : "
        f"{labels.tolist()}"
    )

    counter = Counter(
        labels.tolist()
    )

    print()
    print(
        "当前 Batch 标签："
    )

    for label, count in sorted(
        counter.items()
    ):

        print(
            f"  "
            f"{LABEL_TO_NAME.get(label, str(label))}: "
            f"{count}"
        )

    print()
    print(
        "DataLoader 测试成功。"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print_environment()

    print()
    print("=" * 70)
    print("路径检查")
    print("=" * 70)

    print()
    print(
        f"PROJECT_ROOT:\n"
        f"{PROJECT_ROOT}"
    )

    print()
    print(
        f"DATA_ROOT:\n"
        f"{DATA_ROOT}"
    )

    print()
    print(
        f"TRAIN_JSON:\n"
        f"{TRAIN_JSON}"
    )

    print()
    print(
        f"TRAIN_IMAGE_DIR:\n"
        f"{TRAIN_IMAGE_DIR}"
    )

    print()
    print(
        f"TEST_JSON:\n"
        f"{TEST_JSON}"
    )

    print()
    print(
        f"TEST_IMAGE_DIR:\n"
        f"{TEST_IMAGE_DIR}"
    )

    if not TRAIN_JSON.exists():

        raise FileNotFoundError(
            "\n找不到训练 JSON：\n"
            f"{TRAIN_JSON}"
        )

    if not TRAIN_IMAGE_DIR.exists():

        raise FileNotFoundError(
            "\n找不到训练图片目录：\n"
            f"{TRAIN_IMAGE_DIR}"
        )

    # ========================================================
    # Final candidate settings
    # ========================================================

    train_loader, test_loader = build_dataloaders(
        batch_size=4,
        image_size=448,
        num_workers=0,
        return_metadata=False,
    )

    print()
    print(
        "Train Dataset Size:"
    )

    print(
        len(train_loader.dataset)
    )

    test_one_batch(
        train_loader
    )

    if test_loader is not None:

        print()
        print(
            "Test Dataset Size:"
        )

        print(
            len(test_loader.dataset)
        )

        test_one_batch(
            test_loader
        )

    print()
    print("=" * 70)
    print("全部完成")
    print("=" * 70)

    print()
    print(
        "正式默认预处理："
    )

    print(
        "Original"
    )

    print(
        "   ↓"
    )

    print(
        "Preserve Aspect Ratio"
    )

    print(
        "   ↓"
    )

    print(
        "Pad To Square"
    )

    print(
        "   ↓"
    )

    print(
        "Resize 448 × 448"
    )

    print(
        "   ↓"
    )

    print(
        "Very Conservative Augmentation"
    )

    print(
        "   ↓"
    )

    print(
        "Tensor + Normalize"
    )

    print()
    print(
        "训练代码使用："
    )

    print()
    print(
        "from dataloader import build_dataloaders"
    )

    print()
    print(
        "train_loader, test_loader = build_dataloaders("
    )

    print(
        "    batch_size=4,"
    )

    print(
        "    image_size=448,"
    )

    print(
        "    num_workers=0"
    )

    print(
        ")"
    )

    print()
    print(
        "for images, labels in train_loader:"
    )

    print(
        "    # images: [B, 3, 448, 448]"
    )

    print(
        "    # labels: [B]"
    )

    print(
        "    ..."
    )

    print()
    print(
        "0 = benign"
    )

    print(
        "1 = malignant"
    )

    print()
    print(
        "默认：image_size=448, batch_size=4"
    )

    print("=" * 70)
