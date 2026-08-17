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
    """
    Extract the numeric FNAC category.

    Examples:
        '2' -> 2
        '6' -> 6
        '6 (papillary thyroid carcinoma)' -> 6
        None -> NaN
    """

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


def fnac_contains_ptc(value):
    """
    Determine whether FNAC explicitly mentions
    papillary thyroid carcinoma.
    """

    if pd.isna(value):
        return False

    text = str(value).lower()

    return (
        "papillary thyroid carcinoma" in text
        or "papillary carcinoma" in text
        or "ptc" in text
    )


def histopathology_is_ptc(value):
    """
    Determine whether histopathology explicitly
    identifies papillary thyroid carcinoma.
    """

    if pd.isna(value):
        return False

    text = str(value).lower()

    return (
        "papillary thyroid carcinoma" in text
        or "papillary carcinoma" in text
    )


# ============================================================
# Build classification labels
# ============================================================

def build_labels(df):

    result = df.copy()

    # --------------------------------------------------------
    # 1. Benign / malignant
    # --------------------------------------------------------
    #
    # Original COCO category:
    #   0 = benign
    #   1 = malignant
    #
    # This mapping is explicitly present in the source JSON.
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
    # 4. PTC vs non-PTC
    # --------------------------------------------------------
    #
    # Primary evidence:
    #   Histopathology, when available.
    #
    # If histopathology is unavailable, an explicit
    # PTC statement in FNAC is used.
    #
    # This preserves cases such as:
    #   Histopathology = Papillary thyroid carcinoma
    #
    # and:
    #   FNAC = '6 (papillary thyroid carcinoma)'
    #   Histopathology = null
    #

    result["histopathology_ptc"] = (
        result["histopathology"]
        .apply(histopathology_is_ptc)
    )

    result["fnac_ptc"] = (
        result["fnac"]
        .apply(fnac_contains_ptc)
    )

    result["ptc"] = (
        result["histopathology_ptc"]
        | result["fnac_ptc"]
    ).astype(int)

    result["ptc_source"] = np.select(
        [
            result["histopathology_ptc"],
            result["fnac_ptc"]
        ],
        [
            "histopathology",
            "fnac"
        ],
        default="non_ptc"
    )

    # --------------------------------------------------------
    # Image-level classification target
    # --------------------------------------------------------
    #
    # Every image from the same patient receives the
    # corresponding patient/nodule label.
    #

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
                "ptc"
            ]
        ]
        .value_counts()
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
        .value_counts(dropna=False)
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
        .value_counts(dropna=False)
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
            .nunique(dropna=False)
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
# Save classification datasets
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
        "ptc_source",

        "fnac_class",
        "tirads_class",

        "histopathology_ptc",
        "fnac_ptc",

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
    # One row per patient
    # --------------------------------------------------------

    patient_columns = [
        "patient_id",
        "split",
        "benign_malignant",
        "benign_malignant_name",
        "ptc",
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

        task_df = df[
            columns
        ].copy()

        task_df.to_csv(
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