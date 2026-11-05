
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sdn.controller.dnn_detector import D1DNNDetector


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

RAW_PATH = "data/raw/dataset1/dataset_sdn.csv"

MEDIANS_PATH = "results/preprocessing/dataset1_medians.pkl"
SCALER_PATH = "results/preprocessing/dataset1_scaler.pkl"

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
    "tot_kbps"
]

PROTOCOL_MAPPING = {
    "ICMP": 0,
    "TCP": 1,
    "UDP": 2
}

TOLERANCE = 1e-6


# ============================================================
# PREPROCESSING HELPER
# ============================================================

def encode_protocol(data):
    """
    Apply the same protocol encoding used during training.
    Supports both string and already-encoded numeric values.
    """

    data = data.copy()

    if "Protocol" not in data.columns:
        raise ValueError("Protocol column is missing.")

    protocol_values = data["Protocol"]

    if protocol_values.dtype == object:
        encoded = (
            protocol_values
            .astype(str)
            .str.strip()
            .str.upper()
            .map(PROTOCOL_MAPPING)
        )

        unknown = protocol_values[encoded.isna()].unique()

        if len(unknown) > 0:
            raise ValueError(
                f"Unknown protocol values found: {unknown}"
            )

        data["Protocol"] = encoded

    return data


def clean_numeric_features(data):
    """
    Convert all model features to numeric and replace
    infinite values with NaN.
    """

    data = data.copy()

    for feature in FEATURES:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("D1 DEPLOYMENT PREPROCESSING DEBUG")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD RAW DATA
    # --------------------------------------------------------

    df = pd.read_csv(RAW_PATH)

    print(f"\nOriginal dataset rows: {len(df):,}")

    df = df.drop_duplicates()

    print(f"Rows after duplicate removal: {len(df):,}")

    # --------------------------------------------------------
    # RECREATE EXACT PREPROCESSING SPLIT
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    print(f"\nTraining rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")

    # --------------------------------------------------------
    # LOAD PREPROCESSING ARTIFACTS
    # --------------------------------------------------------

    medians = joblib.load(MEDIANS_PATH)
    scaler = joblib.load(SCALER_PATH)

    # --------------------------------------------------------
    # SELECT ONE RAW TEST RECORD
    # --------------------------------------------------------

    raw_record = test_df.iloc[0].copy()

    print("\nSelected raw test record:")
    print(f"Label: {raw_record['label']}")
    print(f"Raw Protocol: {raw_record['Protocol']}")

    # --------------------------------------------------------
    # OFFICIAL TRAINING PREPROCESSING
    # --------------------------------------------------------

    official = pd.DataFrame(
        [raw_record[FEATURES].to_dict()],
        columns=FEATURES
    )

    # Protocol encoding
    official = encode_protocol(official)

    # Numeric conversion and cleaning
    official = clean_numeric_features(official)

    # Training medians
    official = official.fillna(medians)

    # Scaling
    official_scaled = scaler.transform(
        official[FEATURES]
    )[0]

    # --------------------------------------------------------
    # DEPLOYMENT PREPROCESSING
    # --------------------------------------------------------

    detector = D1DNNDetector()

    # Build deployment input
    deployment_data = {}

    for feature in FEATURES:
        deployment_data[feature] = raw_record.get(
            feature,
            np.nan
        )

    deployment = pd.DataFrame(
        [deployment_data],
        columns=FEATURES
    )

    # IMPORTANT:
    # Encode Protocol BEFORE numeric conversion.
    deployment = encode_protocol(deployment)

    # Numeric conversion and cleaning
    deployment = clean_numeric_features(deployment)

    # Apply detector's saved training medians
    for feature in FEATURES:

        if pd.isna(deployment.at[0, feature]):

            if feature not in detector.medians:
                raise KeyError(
                    f"Median not found for feature: {feature}"
                )

            deployment.at[0, feature] = (
                detector.medians[feature]
            )

    # Scaling using detector's scaler
    deployment_scaled = detector.scaler.transform(
        deployment[FEATURES]
    )[0]

    # --------------------------------------------------------
    # COMPARE SCALED FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("FEATURE-BY-FEATURE PREPROCESSING COMPARISON")
    print("=" * 100)

    differences = []
    max_difference = 0.0

    for i, feature in enumerate(FEATURES):

        official_value = official_scaled[i]
        deployment_value = deployment_scaled[i]

        difference = abs(
            official_value - deployment_value
        )

        max_difference = max(
            max_difference,
            difference
        )

        status = (
            "MATCH"
            if difference <= TOLERANCE
            else "MISMATCH"
        )

        print(
            f"{feature:15s} | "
            f"Official: {official_value:14.8f} | "
            f"Deployment: {deployment_value:14.8f} | "
            f"Diff: {difference:.10f} | "
            f"{status}"
        )

        if difference > TOLERANCE:
            differences.append(feature)

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("PREPROCESSING VALIDATION SUMMARY")
    print("=" * 100)

    print(f"Total features checked: {len(FEATURES)}")
    print(f"Matching features: {len(FEATURES) - len(differences)}")
    print(f"Mismatching features: {len(differences)}")
    print(f"Maximum absolute difference: {max_difference:.12f}")

    if differences:

        print("\nMISMATCH FOUND")

        print("Different features:")

        for feature in differences:
            print(f" - {feature}")

    else:

        print("\nNO PREPROCESSING DIFFERENCE FOUND.")
        print("All 19 features match within the specified tolerance.")

    print("=" * 100)


if __name__ == "__main__":
    main()
