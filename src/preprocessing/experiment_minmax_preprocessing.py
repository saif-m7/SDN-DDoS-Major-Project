
"""
Dataset 1 - MinMaxScaler Preprocessing Experiment

- Does not train any model.
- Does not overwrite baseline processed datasets.
- Fits imputation medians and MinMaxScaler using training data only.
- Uses a stratified 70/15/15 split with random_state=42.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    BASE_DIR / "data" / "raw" / "dataset1" / "dataset_sdn.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "experiments" / "dataset1_minmax"
)

ARTIFACT_DIR = (
    BASE_DIR / "results" / "experiments" / "dataset1_minmax"
)

RANDOM_STATE = 42
TEST_SIZE = 0.30
VALIDATION_TEST_SPLIT = 0.50

LABEL_COLUMN = "label"
TIME_COLUMN = "dt"

FEATURE_COLUMNS = [
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

PROTOCOL_MAPPING = {
    "ICMP": 0,
    "TCP": 1,
    "UDP": 2,
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_csv(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def check_numeric_data(dataframe, split_name):
    features = dataframe[FEATURE_COLUMNS]

    if features.isnull().any().any():
        raise ValueError(
            f"{split_name} contains missing feature values."
        )

    values = features.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{split_name} contains infinite or non-finite values."
        )


def prepare_output(split_df, split_times):
    output_df = split_df.copy()

    if split_times is not None:
        output_df[TIME_COLUMN] = split_times.loc[output_df.index]

        output_df = output_df.sort_values(
            by=TIME_COLUMN,
            kind="stable",
        )

    output_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    if TIME_COLUMN in output_df.columns:
        output_columns.append(TIME_COLUMN)

    return output_df[output_columns].reset_index(drop=True)


# ============================================================
# 3. MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("DATASET 1 - MINMAXSCALER PREPROCESSING")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    print(f"\nLoading raw dataset:\n{RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    timestamps = (
        df[TIME_COLUMN].copy()
        if TIME_COLUMN in df.columns
        else None
    )

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    original_row_count = len(df)

    df = df.drop_duplicates().copy()

    duplicates_removed = original_row_count - len(df)

    if timestamps is not None:
        timestamps = timestamps.loc[df.index]

    print(f"Exact duplicate rows removed: {duplicates_removed:,}")
    print(f"Rows after duplicate removal: {len(df):,}")

    # --------------------------------------------------------
    # Encode protocol
    # --------------------------------------------------------

    protocol_values = (
        df["Protocol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unknown_protocols = (
        set(protocol_values.unique())
        - set(PROTOCOL_MAPPING.keys())
    )

    if unknown_protocols:
        raise ValueError(
            f"Unexpected protocol values: {sorted(unknown_protocols)}"
        )

    df["Protocol"] = protocol_values.map(PROTOCOL_MAPPING)

    # --------------------------------------------------------
    # Validate binary labels
    # --------------------------------------------------------

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="raise",
    )

    if df[LABEL_COLUMN].isnull().any():
        raise ValueError("The label column contains missing values.")

    labels = set(df[LABEL_COLUMN].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Expected labels 0 and 1, but found: {sorted(labels)}"
        )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # --------------------------------------------------------
    # Prepare model features
    # --------------------------------------------------------

    model_df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()

    # Explicitly convert features to numeric values.
    for column in FEATURE_COLUMNS:
        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce",
        )

    # Treat infinities as missing values for median imputation.
    model_df[FEATURE_COLUMNS] = (
        model_df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
    )

    print("\nLabel distribution before splitting:")
    print(model_df[LABEL_COLUMN].value_counts().sort_index())

    # --------------------------------------------------------
    # Split data: 70% train, 15% validation, 15% test
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        model_df,
        test_size=TEST_SIZE,
        stratify=model_df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=VALIDATION_TEST_SPLIT,
        stratify=temp_df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )

    train_times = (
        timestamps.loc[train_df.index]
        if timestamps is not None
        else None
    )

    validation_times = (
        timestamps.loc[validation_df.index]
        if timestamps is not None
        else None
    )

    test_times = (
        timestamps.loc[test_df.index]
        if timestamps is not None
        else None
    )

    print("\nSplit sizes:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")
    print(f"Test:       {len(test_df):,}")

    # --------------------------------------------------------
    # Median imputation: fit on training data only
    # --------------------------------------------------------

    training_medians = train_df[FEATURE_COLUMNS].median()

    if training_medians.isnull().any():
        invalid_features = training_medians[
            training_medians.isnull()
        ].index.tolist()

        raise ValueError(
            f"Cannot calculate medians for: {invalid_features}"
        )

    for split_df in (train_df, validation_df, test_df):
        split_df[FEATURE_COLUMNS] = (
            split_df[FEATURE_COLUMNS]
            .fillna(training_medians)
            .astype(np.float64)
        )

    # --------------------------------------------------------
    # MinMaxScaler: fit on training data only
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    train_df[FEATURE_COLUMNS] = scaler.fit_transform(
        train_df[FEATURE_COLUMNS]
    )

    validation_df[FEATURE_COLUMNS] = scaler.transform(
        validation_df[FEATURE_COLUMNS]
    )

    test_df[FEATURE_COLUMNS] = scaler.transform(
        test_df[FEATURE_COLUMNS]
    )

    # --------------------------------------------------------
    # Validate transformed features
    # --------------------------------------------------------

    check_numeric_data(train_df, "Training data")
    check_numeric_data(validation_df, "Validation data")
    check_numeric_data(test_df, "Test data")

    # --------------------------------------------------------
    # Restore timestamps and sort output by dt
    # --------------------------------------------------------

    train_output = prepare_output(train_df, train_times)

    validation_output = prepare_output(
        validation_df,
        validation_times,
    )

    test_output = prepare_output(test_df, test_times)

    # --------------------------------------------------------
    # Save independent processed datasets
    # --------------------------------------------------------

    save_csv(train_output, OUTPUT_DIR / "train.csv")
    save_csv(validation_output, OUTPUT_DIR / "validation.csv")
    save_csv(test_output, OUTPUT_DIR / "test.csv")

    # --------------------------------------------------------
    # Save preprocessing artifacts
    # --------------------------------------------------------

    scaler_path = ARTIFACT_DIR / "dataset1_minmax_scaler.pkl"
    medians_path = ARTIFACT_DIR / "dataset1_medians.pkl"

    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    with open(medians_path, "wb") as file:
        pickle.dump(training_medians, file)

    print(f"Saved scaler: {scaler_path}")
    print(f"Saved medians: {medians_path}")

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": "Dataset 1",
        "preprocessing": "MinMaxScaler",
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "timestamp_column": (
            TIME_COLUMN if timestamps is not None else None
        ),
        "protocol_mapping": PROTOCOL_MAPPING,
        "duplicate_rows_removed": int(duplicates_removed),
        "random_state": RANDOM_STATE,
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "imputation": "Training-set feature medians",
        "scaler_fit_on": "Training features only",
        "training_rows": int(len(train_output)),
        "validation_rows": int(len(validation_output)),
        "test_rows": int(len(test_output)),
        "training_label_distribution": {
            str(k): int(v)
            for k, v in train_output[LABEL_COLUMN]
            .value_counts().sort_index().items()
        },
        "validation_label_distribution": {
            str(k): int(v)
            for k, v in validation_output[LABEL_COLUMN]
            .value_counts().sort_index().items()
        },
        "test_label_distribution": {
            str(k): int(v)
            for k, v in test_output[LABEL_COLUMN]
            .value_counts().sort_index().items()
        },
    }

    metadata_path = ARTIFACT_DIR / "preprocessing_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Saved metadata: {metadata_path}")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MINMAXSCALER PREPROCESSING COMPLETED")
    print("=" * 65)

    print(f"Processed data: {OUTPUT_DIR}")
    print(f"Artifacts:      {ARTIFACT_DIR}")

    print(f"Train rows:      {len(train_output):,}")
    print(f"Validation rows: {len(validation_output):,}")
    print(f"Test rows:       {len(test_output):,}")

    print("\nNo DNN training was performed.")


if __name__ == "__main__":
    main()
