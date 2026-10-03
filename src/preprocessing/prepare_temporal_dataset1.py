import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


RAW_PATH = Path("data/raw/dataset1/dataset_sdn.csv")
OUTPUT_DIR = Path("data/processed/dataset1_temporal")

SEQUENCE_LENGTH = 10
MAX_GAP = 60

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
    # "Pairflow",  # Excluded due to severe temporal distribution shift
    "Protocol",
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
]


def create_sequences(X, y, dt, sequence_length, max_gap):
    """
    Create real temporal sequences.

    Each sequence contains consecutive observations in time.
    The label corresponds to the final observation.
    """

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length - 1, len(X)):

        start = i - sequence_length + 1

        time_window = dt[start:i + 1]

        # Make sure the complete sequence is temporally continuous.
        if np.max(np.diff(time_window)) > max_gap:
            continue

        X_sequences.append(X[start:i + 1])
        y_sequences.append(y[i])

    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_sequences)


def main():

    print("=" * 70)
    print("TEMPORAL DATASET PREPARATION - DATASET 1")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load raw dataset
    # ---------------------------------------------------------
    df = pd.read_csv(RAW_PATH)

    print(f"\nRaw dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Validate columns
    # ---------------------------------------------------------
    required_columns = FEATURES + ["dt", "label"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 3. Remove exact duplicates
    # ---------------------------------------------------------
    before = len(df)

    df = df.drop_duplicates().copy()

    print(f"Rows before duplicate removal: {before}")
    print(f"Rows after duplicate removal : {len(df)}")
    print(f"Duplicates removed            : {before - len(df)}")

    # ---------------------------------------------------------
    # 4. Encode protocol
    # ---------------------------------------------------------
    protocol_mapping = {
        "ICMP": 0,
        "TCP": 1,
        "UDP": 2
    }

    df["Protocol"] = df["Protocol"].map(protocol_mapping)

    if df["Protocol"].isna().any():
        raise ValueError("Unknown protocol values found.")

    # ---------------------------------------------------------
    # 5. Convert dt to numeric
    # ---------------------------------------------------------
    df["dt"] = pd.to_numeric(df["dt"], errors="coerce")

    if df["dt"].isna().any():
        raise ValueError("Invalid dt values found.")

    # ---------------------------------------------------------
    # 6. Sort chronologically
    # ---------------------------------------------------------
    df = df.sort_values("dt").reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Convert labels
    # ---------------------------------------------------------
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    # ---------------------------------------------------------
    # 8. Time-aware split
    # ---------------------------------------------------------
    unique_dt = np.sort(df["dt"].unique())

    n_timestamps = len(unique_dt)

    train_end = int(n_timestamps * 0.70)
    validation_end = int(n_timestamps * 0.85)

    train_dt = unique_dt[:train_end]
    validation_dt = unique_dt[train_end:validation_end]
    test_dt = unique_dt[validation_end:]

    train_df = df[df["dt"].isin(train_dt)].copy()
    validation_df = df[df["dt"].isin(validation_dt)].copy()
    test_df = df[df["dt"].isin(test_dt)].copy()

    print("\nTime-aware split:")
    print(
        f"Train      : {len(train_df)} rows "
        f"({train_df['dt'].min()} -> {train_df['dt'].max()})"
    )
    print(
        f"Validation : {len(validation_df)} rows "
        f"({validation_df['dt'].min()} -> {validation_df['dt'].max()})"
    )
    print(
        f"Test       : {len(test_df)} rows "
        f"({test_df['dt'].min()} -> {test_df['dt'].max()})"
    )

    # ---------------------------------------------------------
    # 9. Extract features and labels
    # ---------------------------------------------------------
    X_train = train_df[FEATURES].copy()
    X_validation = validation_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df["label"].values
    y_validation = validation_df["label"].values
    y_test = test_df["label"].values

    dt_train = train_df["dt"].values
    dt_validation = validation_df["dt"].values
    dt_test = test_df["dt"].values

    # ---------------------------------------------------------
    # 10. Median imputation
    #     Fit ONLY on training data
    # ---------------------------------------------------------
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_validation = imputer.transform(X_validation)
    X_test = imputer.transform(X_test)

    # ---------------------------------------------------------
    # 11. Standardization
    #     Fit ONLY on training data
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 12. Create temporal sequences
    # ---------------------------------------------------------
    X_train_seq, y_train_seq = create_sequences(
        X_train,
        y_train,
        dt_train,
        SEQUENCE_LENGTH,
        MAX_GAP
    )

    X_validation_seq, y_validation_seq = create_sequences(
        X_validation,
        y_validation,
        dt_validation,
        SEQUENCE_LENGTH,
        MAX_GAP
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test,
        y_test,
        dt_test,
        SEQUENCE_LENGTH,
        MAX_GAP
    )

    # ---------------------------------------------------------
    # 13. Save output
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X_train_seq)
    np.save(OUTPUT_DIR / "y_train.npy", y_train_seq)

    np.save(OUTPUT_DIR / "X_validation.npy", X_validation_seq)
    np.save(OUTPUT_DIR / "y_validation.npy", y_validation_seq)

    np.save(OUTPUT_DIR / "X_test.npy", X_test_seq)
    np.save(OUTPUT_DIR / "y_test.npy", y_test_seq)

    # ---------------------------------------------------------
    # 14. Save metadata
    # ---------------------------------------------------------
    metadata = {
        "dataset": "Dataset 1",
        "sequence_length": SEQUENCE_LENGTH,
        "max_allowed_gap_seconds": MAX_GAP,
        "num_features": len(FEATURES),
        "features": FEATURES,
        "excluded_features": {
            "Pairflow": (
                "Excluded because its distribution changes from "
                "binary variation in the training period to a "
                "constant value of 1 in validation and test periods."
            ),
            "src": "Raw source address excluded.",
            "dst": "Raw destination address excluded."
        },
        "split_method": "Chronological time-aware split",
        "split_ratio": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },
        "duplicate_removal": True,
        "imputation": "Median, fitted on training data only",
        "scaling": "StandardScaler, fitted on training data only",
        "shapes": {
            "X_train": list(X_train_seq.shape),
            "y_train": list(y_train_seq.shape),
            "X_validation": list(X_validation_seq.shape),
            "y_validation": list(y_validation_seq.shape),
            "X_test": list(X_test_seq.shape),
            "y_test": list(y_test_seq.shape)
        }
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    # ---------------------------------------------------------
    # 15. Print results
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEMPORAL SEQUENCE RESULTS")
    print("=" * 70)

    print(f"\nNumber of features: {len(FEATURES)}")
    print(f"Sequence length   : {SEQUENCE_LENGTH}")
    print(f"Maximum gap       : {MAX_GAP} seconds")

    print("\nSequence shapes:")
    print(f"Train      : X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(
        f"Validation : "
        f"X={X_validation_seq.shape}, y={y_validation_seq.shape}"
    )
    print(f"Test       : X={X_test_seq.shape}, y={y_test_seq.shape}")

    print("\nFeatures used:")
    for i, feature in enumerate(FEATURES, start=1):
        print(f"{i:2}. {feature}")

    print("\nPairflow: EXCLUDED")

    print("\nSaved to:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("TEMPORAL DATASET PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()