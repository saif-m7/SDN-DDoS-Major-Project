import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

RAW_PATH = "data/raw/dataset1/dataset_sdn.csv"
OUTPUT_DIR = "data/processed/dataset1"

TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_dataset1():

    print("=" * 70)
    print("DATASET 1 PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD RAW DATA
    # --------------------------------------------------------

    print("\nLoading raw Dataset 1...")

    df = pd.read_csv(RAW_PATH)

    print(f"Raw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. BASIC INFORMATION
    # --------------------------------------------------------

    print("\nOriginal columns:")
    print(df.columns.tolist())

    print("\nOriginal label distribution:")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))

    # --------------------------------------------------------
    # 3. CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
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
        "dt",
        "src",
        "dst",
        "label"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 4. REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nExact duplicate rows:", duplicate_count)

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    print(
        "Shape after duplicate removal:",
        df.shape
    )

    # --------------------------------------------------------
    # 5. CHECK LABELS
    # --------------------------------------------------------

    print("\nLabel distribution after duplicate removal:")

    print(df["label"].value_counts())

    print(
        df["label"].value_counts(normalize=True)
    )

    # Ensure labels are integer 0/1
    df["label"] = df["label"].astype(int)

    unique_labels = sorted(df["label"].unique())

    if unique_labels != [0, 1]:
        raise ValueError(
            f"Unexpected labels found: {unique_labels}"
        )

    # --------------------------------------------------------
    # 6. HANDLE PROTOCOL
    # --------------------------------------------------------

    print("\nProtocol values before encoding:")

    print(df["Protocol"].value_counts())

    protocol_mapping = {
        "ICMP": 0,
        "TCP": 1,
        "UDP": 2
    }

    df["Protocol"] = df["Protocol"].map(protocol_mapping)

    if df["Protocol"].isna().any():

        unknown_protocols = df.loc[
            df["Protocol"].isna(),
            "Protocol"
        ]

        raise ValueError(
            "Unknown protocol values found."
        )

    print("\nProtocol encoded as:")
    print(protocol_mapping)

    # --------------------------------------------------------
    # 7. HANDLE MISSING VALUES
    # --------------------------------------------------------

    print("\nMissing values before handling:")

    missing_before = df.isna().sum()

    print(
        missing_before[
            missing_before > 0
        ]
    )

    # Numeric columns that will be used as model features
    numeric_features = [
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
        "port_no",
        "tx_bytes",
        "rx_bytes",
        "tx_kbps",
        "rx_kbps",
        "tot_kbps",
        "Protocol"
    ]

    # Use median imputation for missing numerical values.
    # IMPORTANT:
    # The median will later be calculated from TRAINING data
    # only to avoid data leakage.
    
    # --------------------------------------------------------
    # 8. KEEP DT FOR TEMPORAL INFORMATION
    # --------------------------------------------------------

    # We don't use dt directly as a neural-network feature.
    # It will be preserved in the processed files so that
    # LSTM/GRU sequences can be constructed later.

    dt_column = df["dt"].copy()

    # --------------------------------------------------------
    # 9. REMOVE RAW IP ADDRESSES
    # --------------------------------------------------------

    # Raw source/destination IPs are excluded because the
    # model should learn traffic behavior rather than memorize
    # specific IP addresses.

    df = df.drop(
        columns=["src", "dst"]
    )

    # --------------------------------------------------------
    # 10. SELECT MODEL FEATURES
    # --------------------------------------------------------

    feature_columns = [
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
        "tot_kbps"
    ]

    target_column = "label"

    # --------------------------------------------------------
    # 11. TEMPORARY DATAFRAME FOR SPLITTING
    # --------------------------------------------------------

    model_df = df[
        feature_columns + [target_column]
    ].copy()

    print("\nModel input features:")
    print(feature_columns)

    print(
        f"\nNumber of model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # 12. TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    print("\nCreating train/validation/test split...")

    train_df, temp_df = train_test_split(
        model_df,
        test_size=0.30,
        stratify=model_df[target_column],
        random_state=RANDOM_SEED
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[target_column],
        random_state=RANDOM_SEED
    )

    print("\nSplit sizes:")

    print(f"Training   : {len(train_df):,}")
    print(f"Validation : {len(validation_df):,}")
    print(f"Test       : {len(test_df):,}")

    # --------------------------------------------------------
    # 13. MEDIAN IMPUTATION
    # --------------------------------------------------------

    print("\nHandling missing values...")

    # Calculate medians ONLY from training data.
    # This prevents validation/test information from
    # influencing preprocessing.

    train_medians = train_df[
        feature_columns
    ].median()

    train_df[feature_columns] = (
        train_df[feature_columns]
        .fillna(train_medians)
    )

    validation_df[feature_columns] = (
        validation_df[feature_columns]
        .fillna(train_medians)
    )

    test_df[feature_columns] = (
        test_df[feature_columns]
        .fillna(train_medians)
    )

    # --------------------------------------------------------
    # 14. CHECK INFINITE VALUES
    # --------------------------------------------------------

    for name, dataset in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]:

        numeric_data = dataset[
            feature_columns
        ].select_dtypes(include=np.number)

        if np.isinf(numeric_data).any().any():

            raise ValueError(
                f"Infinite values found in {name} dataset."
            )

    # --------------------------------------------------------
    # 15. STANDARDIZATION
    # --------------------------------------------------------

    print("\nStandardizing numerical features...")

    scaler = StandardScaler()

    train_df[feature_columns] = scaler.fit_transform(
        train_df[feature_columns]
    )

    validation_df[feature_columns] = scaler.transform(
        validation_df[feature_columns]
    )

    test_df[feature_columns] = scaler.transform(
        test_df[feature_columns]
    )

    # --------------------------------------------------------
    # 16. RESTORE DT
    # --------------------------------------------------------

    # For now, dt is kept separately for future temporal
    # sequence construction.
    #
    # We don't include it in the model feature matrix.

    # Since the train/validation/test split shuffled rows,
    # we need to preserve dt corresponding to each split.
    #
    # Re-create the split indices using the original model_df.

    train_indices, temp_indices = train_test_split(
        model_df.index,
        test_size=0.30,
        stratify=model_df[target_column],
        random_state=RANDOM_SEED
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=model_df.loc[
            temp_indices,
            target_column
        ],
        random_state=RANDOM_SEED
    )

    train_dt = dt_column.loc[train_indices].values
    validation_dt = dt_column.loc[val_indices].values
    test_dt = dt_column.loc[test_indices].values

    train_df.insert(
        0,
        "dt",
        train_dt
    )

    validation_df.insert(
        0,
        "dt",
        validation_dt
    )

    test_df.insert(
        0,
        "dt",
        test_dt
    )

    # --------------------------------------------------------
    # 17. SORT TEMPORALLY
    # --------------------------------------------------------

    train_df = train_df.sort_values(
        "dt"
    ).reset_index(drop=True)

    validation_df = validation_df.sort_values(
        "dt"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        "dt"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 18. CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 19. SAVE PROCESSED DATA
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation_df.to_csv(
        VAL_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    # --------------------------------------------------------
    # 20. FINAL VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET CHECK")
    print("=" * 70)

    for name, dataset in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df)
    ]:

        print(f"\n{name}")

        print(
            "Shape:",
            dataset.shape
        )

        print(
            "Missing values:",
            dataset.isna().sum().sum()
        )

        numeric_data = dataset[
            feature_columns
        ].select_dtypes(include=np.number)

        print(
            "Infinite values:",
            np.isinf(numeric_data).sum().sum()
        )

        print(
            "Label distribution:"
        )

        print(
            dataset["label"].value_counts(
                normalize=True
            ).sort_index()
        )

    # --------------------------------------------------------
    # 21. OUTPUT INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print("\nSaved files:")

    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)

    print("\nFinal model features:")

    for i, feature in enumerate(
        feature_columns,
        start=1
    ):
        print(f"{i:2}. {feature}")

    print("\nDT is preserved for future LSTM/GRU sequence construction.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    preprocess_dataset1()