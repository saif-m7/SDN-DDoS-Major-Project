import os
import pandas as pd

from src.inference.d1_detector import detect_ddos
from src.inference.d2_classifier import classify_traffic


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\PROJECTS\SDN-DDOS-Major-Project"


# ============================================================
# TWO-STAGE DDoS DETECTION PIPELINE
# ============================================================

def detect_and_classify(d1_data, d2_data=None):
    """
    Two-stage DDoS detection pipeline.

    Stage 1:
        Dataset 1 DNN
        NORMAL / MALICIOUS

    Stage 2:
        Dataset 2 DNN
        Six-class traffic / DDoS classification

    Parameters
    ----------
    d1_data : pandas.DataFrame
        Raw Dataset 1 traffic data.

    d2_data : pandas.DataFrame, optional
        Dataset 2 traffic data.
        Required only when D1 detects malicious traffic.

    Returns
    -------
    list
        Combined prediction results.
    """

    # ========================================================
    # STAGE 1: D1 DDoS DETECTION
    # ========================================================

    d1_results = detect_ddos(
        d1_data
    )

    final_results = []

    for index, d1_result in enumerate(
        d1_results
    ):

        # ----------------------------------------------------
        # NORMAL TRAFFIC
        # ----------------------------------------------------

        if d1_result["prediction"] == "NORMAL":

            final_results.append({
                "detection": "NORMAL",
                "d1_prediction": "NORMAL",
                "d1_class": d1_result["class"],
                "d1_confidence": d1_result[
                    "normal_probability"
                ],
                "d2_prediction": None,
                "d2_class": None,
                "d2_confidence": None
            })

            continue

        # ----------------------------------------------------
        # MALICIOUS TRAFFIC
        # ----------------------------------------------------

        result = {
            "detection": "MALICIOUS",
            "d1_prediction": "MALICIOUS",
            "d1_class": d1_result["class"],
            "d1_confidence": d1_result[
                "malicious_probability"
            ],
            "d2_prediction": None,
            "d2_class": None,
            "d2_confidence": None
        }

        # ----------------------------------------------------
        # D2 DATA REQUIRED
        # ----------------------------------------------------

        if d2_data is None:

            result["d2_prediction"] = (
                "D2_DATA_REQUIRED"
            )

            final_results.append(
                result
            )

            continue

        # ----------------------------------------------------
        # CHECK D2 RECORD
        # ----------------------------------------------------

        if index >= len(d2_data):

            raise ValueError(
                "D1 and D2 data contain different "
                "numbers of records."
            )

        d2_record = d2_data.iloc[
            index:index + 1
        ]

        # ----------------------------------------------------
        # STAGE 2: D2 CLASSIFICATION
        # ----------------------------------------------------

        d2_result = classify_traffic(
            d2_record
        )[0]

        result["d2_prediction"] = (
            d2_result["prediction"]
        )

        result["d2_class"] = (
            d2_result["class"]
        )

        result["d2_confidence"] = (
            d2_result["confidence"]
        )

        final_results.append(
            result
        )

    return final_results


# ============================================================
# SINGLE TRAFFIC RECORD
# ============================================================

def detect_single_traffic(
    d1_record,
    d2_record=None
):
    """
    Run the complete two-stage pipeline
    for one traffic record.
    """

    d1_df = pd.DataFrame(
        [d1_record]
    )

    d2_df = None

    if d2_record is not None:

        d2_df = pd.DataFrame(
            [d2_record]
        )

    results = detect_and_classify(
        d1_df,
        d2_df
    )

    return results[0]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("TWO-STAGE DDoS DETECTION PIPELINE")
    print("=" * 70)

    # ========================================================
    # LOAD RAW DATASET 1
    # ========================================================

    d1_raw_path = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "dataset1",
        "dataset_sdn.csv"
    )

    d1_raw_df = pd.read_csv(
        d1_raw_path
    )

    # --------------------------------------------------------
    # Validate D1 dataset
    # --------------------------------------------------------

    if d1_raw_df.empty:

        raise ValueError(
            "Dataset 1 raw data is empty."
        )

    if "label" not in d1_raw_df.columns:

        raise ValueError(
            "Dataset 1 raw data does not contain "
            "the label column."
        )

    # ========================================================
    # SELECT MALICIOUS RECORDS
    # ========================================================

    print(
        "\nSearching for a correctly detected "
        "malicious Dataset 1 sample..."
    )

    malicious_df = d1_raw_df[
        d1_raw_df["label"] == 1
    ].copy()

    if malicious_df.empty:

        raise ValueError(
            "No malicious records found in "
            "Dataset 1 raw data."
        )

    print(
        f"Malicious records available: "
        f"{len(malicious_df)}"
    )

    # ========================================================
    # RUN D1 DETECTOR
    # ========================================================

    d1_predictions = detect_ddos(
        malicious_df
    )

    # ========================================================
    # FIND CORRECTLY DETECTED MALICIOUS RECORD
    # ========================================================

    selected_index = None
    selected_prediction = None

    for index, prediction in enumerate(
        d1_predictions
    ):

        if prediction["prediction"] == "MALICIOUS":

            selected_index = index
            selected_prediction = prediction

            break

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if selected_index is None:

        raise RuntimeError(
            "No correctly detected malicious "
            "Dataset 1 sample was found."
        )

    # ========================================================
    # SELECT D1 SAMPLE
    # ========================================================

    d1_sample = malicious_df.iloc[
        selected_index:selected_index + 1
    ].copy()

    print("\nSelected D1 test record")
    print("-" * 40)

    print(
        f"Ground Truth      : "
        f"{d1_sample['label'].iloc[0]}"
    )

    print(
        f"D1 Prediction     : "
        f"{selected_prediction['prediction']}"
    )

    print(
        f"D1 Confidence     : "
        f"{selected_prediction['malicious_probability']:.4f}"
    )

    # ========================================================
    # LOAD DATASET 2 TEST DATA
    # ========================================================

    d2_test_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "dataset2",
        "test.csv"
    )

    d2_test_df = pd.read_csv(
        d2_test_path
    )

    # --------------------------------------------------------
    # Validate D2 dataset
    # --------------------------------------------------------

    if d2_test_df.empty:

        raise ValueError(
            "Dataset 2 test data is empty."
        )

    # ========================================================
    # SELECT ONE D2 RECORD
    # ========================================================

    d2_sample = d2_test_df.iloc[
        0:1
    ].copy()

    # ========================================================
    # RUN TWO-STAGE PIPELINE
    # ========================================================

    results = detect_and_classify(
        d1_sample,
        d2_sample
    )

    result = results[0]

    # ========================================================
    # DISPLAY FINAL RESULT
    # ========================================================

    print("\nFinal Prediction")
    print("-" * 40)

    print(
        f"D1 Detection      : "
        f"{result['d1_prediction']}"
    )

    print(
        f"D1 Confidence     : "
        f"{result['d1_confidence']:.4f}"
    )

    if result["d2_prediction"] is not None:

        print(
            f"D2 Classification : "
            f"{result['d2_prediction']}"
        )

        print(
            f"D2 Confidence     : "
            f"{result['d2_confidence']:.4f}"
        )

    else:

        print(
            "D2 Classification : "
            "Not required"
        )

    # ========================================================
    # PIPELINE STATUS
    # ========================================================

    print("\n" + "=" * 70)
    print("MALICIOUS PIPELINE TEST COMPLETED")
    print("=" * 70)