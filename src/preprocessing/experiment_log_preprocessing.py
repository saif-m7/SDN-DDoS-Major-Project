
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


SEED = 42
LABEL_COLUMN = "label"
TIME_COLUMN = "dt"

FEATURES = [
    "switch",
    "pktcount",
    "bytecount",
    "dur",
    "dur_nsec",
    "tot_dur",
    "flows",
    "packetins",
    "pktperflow",
    "byteperflow",
    "pktrate",
    "Pairflow",
    "Protocol",
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
]


def main():
    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    root = Path(__file__).resolve().parents[2]

    input_path = (
        root / "data" / "raw" / "dataset1" / "dataset_sdn.csv"
    )

    output_dir = (
        root
        / "data"
        / "processed"
        / "experiments"
        / "dataset1_log"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = (
        root / "results" / "experiments" / "dataset1_log"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    print("=" * 65)
    print("DATASET 1 - SHIFTED LOG TRANSFORMATION PREPROCESSING")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(input_path)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    required = FEATURES + [LABEL_COLUMN]
    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Remove exact duplicate rows.
    original_rows = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    print(f"Exact duplicate rows removed: {duplicates_removed:,}")
    print(f"Rows after duplicate removal: {len(df):,}")

    # Validate binary labels.
    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN], errors="raise"
    ).astype("int32")

    if not set(df[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0 and 1; found "
            f"{sorted(df[LABEL_COLUMN].unique())}"
        )

    # --------------------------------------------------------
    # PROTOCOL ENCODING
    # --------------------------------------------------------

    print("\nOriginal Protocol values:")
    print(df["Protocol"].value_counts(dropna=False).head(20))

    protocol_values = (
        df["Protocol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    protocol_map = {
        "ICMP": 0,
        "TCP": 1,
        "UDP": 2,
    }

    mapped_protocol = protocol_values.map(protocol_map)

    # Also support numeric protocol values.
    numeric_protocol = pd.to_numeric(
        protocol_values, errors="coerce"
    )

    df["Protocol"] = mapped_protocol.fillna(numeric_protocol)

    if df["Protocol"].isna().any():
        unknown_values = (
            protocol_values[df["Protocol"].isna()]
            .value_counts(dropna=False)
            .to_dict()
        )

        raise ValueError(
            "Unknown or missing Protocol values remain: "
            f"{unknown_values}"
        )

    print("\nEncoded Protocol distribution:")
    print(df["Protocol"].value_counts(dropna=False))

    # Convert remaining features to numeric.
    for col in FEATURES:
        if col != "Protocol":
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Replace infinite values with NaN.
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    # Convert every model feature to float64 now.
    # This prevents pandas incompatible-dtype warnings
    # during imputation and log transformation.
    df.loc[:, FEATURES] = df[FEATURES].astype("float64")

    # --------------------------------------------------------
    # STRATIFIED 70/15/15 SPLIT
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df[LABEL_COLUMN],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df[LABEL_COLUMN],
    )

    # Ensure the feature columns retain floating-point dtypes.
    for split_df in (train_df, val_df, test_df):
        split_df.loc[:, FEATURES] = (
            split_df[FEATURES].astype("float64")
        )

    print(f"\nTraining rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")

    # --------------------------------------------------------
    # MEDIAN IMPUTATION
    # Fit medians on training data only.
    # --------------------------------------------------------

    medians = train_df[FEATURES].median()

    all_missing = medians[medians.isna()].index.tolist()

    if all_missing:
        raise ValueError(
            "These features have no valid training values: "
            f"{all_missing}"
        )

    for split_df in (train_df, val_df, test_df):
        split_df.loc[:, FEATURES] = (
            split_df[FEATURES]
            .fillna(medians)
            .astype("float64")
        )

    # --------------------------------------------------------
    # SHIFTED LOG TRANSFORMATION
    #
    # shift = max(0, -minimum_training_value)
    # transformed = log1p(value + shift)
    #
    # Shifts are derived from training data only.
    # --------------------------------------------------------

    train_values = train_df[FEATURES].astype("float64")
    shifts = {}

    for feature in FEATURES:
        minimum = float(train_values[feature].min())
        shifts[feature] = max(0.0, -minimum)

    for split_name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        # Work on float64 feature columns to avoid dtype warnings.
        split_df.loc[:, FEATURES] = (
            split_df[FEATURES].astype("float64")
        )

        for feature in FEATURES:
            shifted_values = (
                split_df[feature] + shifts[feature]
            )

            if (shifted_values <= -1).any():
                raise ValueError(
                    f"{split_name} contains values <= -1 after "
                    f"the training-derived shift for '{feature}'. "
                    "Inspect the feature range before proceeding."
                )

            split_df[feature] = np.log1p(
                shifted_values
            )

    print("\nShifted log transformation applied.")
    print("Features with non-zero shifts:")

    shifted_features = {
        feature: shift
        for feature, shift in shifts.items()
        if shift > 0
    }

    if shifted_features:
        for feature, shift in shifted_features.items():
            print(f"  {feature}: shift = {shift}")
    else:
        print("  None")

    # --------------------------------------------------------
    # VALIDATE TRANSFORMED DATA
    # --------------------------------------------------------

    for split_name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        values = split_df[FEATURES].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{split_name} contains non-finite "
                "transformed feature values."
            )

    # --------------------------------------------------------
    # SAVE PROCESSED SPLITS
    # --------------------------------------------------------

    for name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        if TIME_COLUMN in split_df.columns:
            split_df = split_df.sort_values(
                TIME_COLUMN, kind="stable"
            )

        output_path = output_dir / f"{name}.csv"
        split_df.to_csv(output_path, index=False)

        print(
            f"Saved {name}: {output_path} "
            f"({len(split_df):,} rows)"
        )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "experiment": "Dataset 1 shifted log transformation",
        "input_file": str(input_path),
        "transformation": "log1p(x + shift)",
        "features": FEATURES,
        "label_column": LABEL_COLUMN,
        "random_state": SEED,
        "split": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "duplicate_rows_removed": duplicates_removed,
        "imputation": "Training-set median per feature",
        "protocol_mapping": protocol_map,
        "feature_shifts": {
            feature: float(shift)
            for feature, shift in shifts.items()
        },
        "shift_method": (
            "max(0, -minimum training value) per feature"
        ),
        "notes": [
            "Protocol is encoded before numeric feature conversion.",
            "Feature columns are converted to float64 before transformation.",
            "Shifts are calculated using training data only.",
            "The same shifts are applied to validation and test data.",
            "This is shifted log transformation, not plain log1p.",
            "Verify split membership before comparing experiments.",
        ],
    }

    metadata_path = (
        metadata_dir / "preprocessing_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nPreprocessing completed successfully.")
    print(f"Metadata saved: {metadata_path}")
    print("No DNN training was performed.")


if __name__ == "__main__":
    main()
