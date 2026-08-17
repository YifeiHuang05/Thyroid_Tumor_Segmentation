from pathlib import Path
import pandas as pd
import numpy as np
import re


# ============================================================
# Paths
# ============================================================

DATASET_ROOT = Path(
    "data/ThyroidXL_clean"
)

MASTER_CSV = (
    DATASET_ROOT
    / "metadata"
    / "master_annotations.csv"
)

OUTPUT_DIR = (
    DATASET_ROOT
    / "classification"
)


# ============================================================
# Label parsing
# ============================================================

def parse_fnac_value(value):
    """Extract numeric FNAC category."""

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    match = re.match(
        r"^\s*(\d+)",
        text
    )

    if match:
        return int(match.group(1))

    return np.nan


def histopathology_is_ptc(value):
    """
    Histopathology-based PTC definition.

    Returns:
        True  = histopathology explicitly indicates PTC
        False = histopathology is present and does not indicate PTC
        NaN   = histopathology unavailable
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if not text:
        return np.nan

    if (
        "papillary thyroid carcinoma" in text
        or "papillary carcinoma" in text
    ):
        return True

    return False


# ============================================================
# Build classification labels
# ============================================================

def build_labels(df):

    result = df.copy()

    # --------------------------------------------------------
    # 1. Benign / malignant
    # --------------------------------------------------------
    #
    # Original annotation:
    # 0 = benign
    # 1 = malignant
    #

    result["benign_malignant"] = (
        result["original_category_id"]
        .astype(int)
    )

    result["benign_malignant_name"] = (
        result["benign_malignant"]
        .map({
            0: "benign",
            1: "malignant"
        })
    )

    # --------------------------------------------------------
    # 2. FNAC
    # --------------------------------------------------------

    result["fnac_class"] = (
        result["fnac"]
        .apply(parse_fnac_value)
    )

    # --------------------------------------------------------
    # 3. TIRADS
    # --------------------------------------------------------

    result["tirads_class"] = (
        pd.to_numeric(
            result["tirads"],
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # 4. PTC
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Official PTC scoring is based ONLY on
    # histopathology.
    #
    # PTC:
    #   histopathology = papillary thyroid carcinoma
    #       -> 1
    #
    # Non-PTC:
    #   histopathology is available but diagnosis
    #   is not PTC
    #       -> 0
    #
    # Unknown:
    #   histopathology unavailable
    #       -> NaN
    #
    # FNAC is NOT used to determine PTC.
    #

    result["ptc"] = (
        result["histopathology"]
        .apply(histopathology_is_ptc)
    )

    result["ptc_source"] = np.where(
        result["ptc"].isna(),
        "missing_histopathology",
        "histopathology"
    )

    # --------------------------------------------------------
    # Explicit PTC label name
    # --------------------------------------------------------

    result["ptc_name"] = (
        result["ptc"]
        .map({
            0.0: "non-PTC",
            1.0: "PTC"
        })
    )

    return result


# ============================================================
# Audit
# ============================================================

def audit_labels(df):

    print("\n" + "=" * 70)
    print("CLASSIFICATION LABEL AUDIT")
    print("=" * 70)

    print(
        f"\nTotal images:   {len(df)}"
    )

    print(
        f"Total patients: "
        f"{df['patient_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Benign / malignant
    # --------------------------------------------------------

    print("\n--- Benign / malignant ---")

    print(
        df[
            [
                "split",
                "benign_malignant_name"
            ]
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # PTC
    # --------------------------------------------------------

    print("\n--- PTC / non-PTC ---")

    print(
        df[
            [
                "split",
                "ptc_name"
            ]
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("\nPTC label source:")

    print(
        df[
            [
                "split",
                "ptc_source"
            ]
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # PTC test-set evaluation population
    # --------------------------------------------------------

    test_ptc = df[
        df["split"] == "test"
    ].copy()

    test_ptc_labeled = test_ptc[
        test_ptc["ptc"].notna()
    ]

    print(
        "\n--- Official PTC test population ---"
    )

    print(
        f"Test images with histopathology: "
        f"{len(test_ptc_labeled)}"
    )

    print(
        f"Test images without histopathology: "
        f"{test_ptc['ptc'].isna().sum()}"
    )

    print("\nOfficial PTC test labels:")

    print(
        test_ptc_labeled[
            "ptc_name"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # FNAC
    # --------------------------------------------------------

    print("\n--- FNAC ---")

    print(
        df[
            [
                "split",
                "fnac_class"
            ]
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # TIRADS
    # --------------------------------------------------------

    print("\n--- TIRADS ---")

    print(
        df[
            [
                "split",
                "tirads_class"
            ]
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # Missing labels
    # --------------------------------------------------------

    print("\n--- Missing labels ---")

    for column in [
        "benign_malignant",
        "ptc",
        "fnac_class",
        "tirads_class"
    ]:

        missing = df[column].isna().sum()

        print(
            f"{column:25s}: "
            f"{missing:5d} missing "
            f"({missing / len(df):.2%})"
        )

    # --------------------------------------------------------
    # Patient-level consistency
    # --------------------------------------------------------

    print("\n--- Patient-level consistency ---")

    for column in [
        "benign_malignant",
        "ptc",
        "fnac_class",
        "tirads_class"
    ]:

        consistency = (
            df
            .groupby("patient_id")[column]
            .nunique(
                dropna=False
            )
        )

        inconsistent = (
            consistency > 1
        ).sum()

        print(
            f"{column:25s}: "
            f"{inconsistent} patients "
            "with inconsistent labels"
        )


# ============================================================
# Save outputs
# ============================================================

def save_outputs(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Complete metadata
    # --------------------------------------------------------

    output_columns = [
        "image_id",
        "image_name",
        "patient_id",
        "split",

        "benign_malignant",
        "benign_malignant_name",

        "ptc",
        "ptc_name",
        "ptc_source",

        "fnac_class",
        "tirads_class",

        "original_category_id",

        "age",
        "gender",
        "fnac",
        "tirads",
        "histopathology",
        "conclusion"
    ]

    output_columns = [
        x for x in output_columns
        if x in df.columns
    ]

    df[
        output_columns
    ].to_csv(
        OUTPUT_DIR
        / "classification_metadata.csv",
        index=False
    )

    # --------------------------------------------------------
    # Patient-level labels
    # --------------------------------------------------------

    patient_columns = [
        "patient_id",
        "split",
        "benign_malignant",
        "benign_malignant_name",
        "ptc",
        "ptc_name",
        "ptc_source",
        "fnac_class",
        "tirads_class",
        "histopathology",
        "fnac",
        "conclusion"
    ]

    patient_columns = [
        x for x in patient_columns
        if x in df.columns
    ]

    patient_df = (
        df[
            patient_columns
        ]
        .drop_duplicates(
            subset=["patient_id"]
        )
        .sort_values(
            ["split", "patient_id"]
        )
    )

    patient_df.to_csv(
        OUTPUT_DIR
        / "patient_classification_labels.csv",
        index=False
    )

    # --------------------------------------------------------
    # Individual task files
    # --------------------------------------------------------

    tasks = {

        "benign_malignant":
            [
                "image_name",
                "patient_id",
                "split",
                "benign_malignant",
                "benign_malignant_name"
            ],

        "ptc":
            [
                "image_name",
                "patient_id",
                "split",
                "ptc",
                "ptc_name",
                "ptc_source"
            ],

        "fnac":
            [
                "image_name",
                "patient_id",
                "split",
                "fnac_class"
            ],

        "tirads":
            [
                "image_name",
                "patient_id",
                "split",
                "tirads_class"
            ]
    }

    for task, columns in tasks.items():

        columns = [
            x for x in columns
            if x in df.columns
        ]

        df[
            columns
        ].to_csv(
            OUTPUT_DIR
            / f"{task}_labels.csv",
            index=False
        )


# ============================================================
# Main
# ============================================================

def main():

    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{MASTER_CSV}"
        )

    print("=" * 70)
    print("PREPARING THYROIDXL CLASSIFICATION TARGETS")
    print("=" * 70)

    print(
        f"\nInput:\n{MASTER_CSV}"
    )

    df = pd.read_csv(
        MASTER_CSV
    )

    print(
        f"\nLoaded {len(df)} image records."
    )

    df = build_labels(df)

    audit_labels(df)

    save_outputs(df)

    print("\n" + "=" * 70)
    print("CLASSIFICATION METADATA READY")
    print("=" * 70)

    print(
        f"""
Output directory:

{OUTPUT_DIR}

Files:

classification_metadata.csv
patient_classification_labels.csv

benign_malignant_labels.csv
ptc_labels.csv
fnac_labels.csv
tirads_labels.csv
"""
    )


if __name__ == "__main__":
    main()