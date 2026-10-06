import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sdn.controller.dnn_detector import D1DNNDetector


RANDOM_SEED = 42

RAW_PATH = "data/raw/dataset1/dataset_sdn.csv"

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


def main():

    print("=" * 70)
    print("D1 DEPLOYMENT PREPROCESSING DEBUG")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    df = pd.read_csv(RAW_PATH)

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Recreate EXACT preprocessing split
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

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    medians = joblib.load(
        "results/preprocessing/dataset1_medians.pkl"
    )

    scaler = joblib.load(
        "results/preprocessing/dataset1_scaler.pkl"
    )

    # --------------------------------------------------------
    # Select ONE raw test record
    # --------------------------------------------------------

    raw_record = test_df.iloc[0].copy()

    print("\nRaw record label:")
    print(raw_record["label"])

    # --------------------------------------------------------
    # OFFICIAL PREPROCESSING
    # --------------------------------------------------------

    official = pd.DataFrame(
        [raw_record[FEATURES].to_dict()],
        columns=FEATURES
    )

    official["Protocol"] = (
        official["Protocol"]
        .astype(str)
        .str.upper()
        .map(PROTOCOL_MAPPING)
    )

    for feature in FEATURES:
        official[feature] = pd.to_numeric(
            official[feature],
            errors="coerce"
        )

    official = official.replace(
        [np.inf, -np.inf],
        np.nan
    )

    official = official.fillna(
        medians
    )

    official_scaled = scaler.transform(
        official[FEATURES]
    )[0]

    # --------------------------------------------------------
    # DEPLOYMENT PREPROCESSING
    # --------------------------------------------------------

    detector = D1DNNDetector()

    deployment_scaled = detector.scaler.transform(
        detector.scaler.feature_names_in_.reshape(1, -1)
    ) if False else None

    # Reproduce exactly what detector.predict() does
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

    for feature in FEATURES:
        deployment[feature] = pd.to_numeric(
            deployment[feature],
            errors="coerce"
        )

    deployment = deployment.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for feature in FEATURES:
        if pd.isna(deployment.at[0, feature]):
            deployment.at[0, feature] = detector.medians[
                feature
            ]

    deployment_scaled = detector.scaler.transform(
        deployment[FEATURES]
    )[0]

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE COMPARISON")
    print("=" * 70)

    differences = []

    for i, feature in enumerate(FEATURES):

        official_value = official_scaled[i]
        deployment_value = deployment_scaled[i]

        difference = abs(
            official_value - deployment_value
        )

        print(
            f"{feature:15s} | "
            f"Official: {official_value:12.6f} | "
            f"Deployment: {deployment_value:12.6f} | "
            f"Diff: {difference:.10f}"
        )

        if difference > 1e-6:
            differences.append(feature)

    print("\n" + "=" * 70)

    if differences:
        print("MISMATCH FOUND")
        print("Different features:")
        for feature in differences:
            print(" -", feature)
    else:
        print("NO PREPROCESSING DIFFERENCE FOUND.")

    print("=" * 70)


if __name__ == "__main__":
    main()
