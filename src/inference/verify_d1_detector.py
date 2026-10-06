import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sdn.controller.dnn_detector import D1DNNDetector


RANDOM_SEED = 42
RAW_PATH = "data/raw/dataset1/dataset_sdn.csv"


def main():

    print("=" * 70)
    print("D1 DNN DETECTOR - EXACT TEST SET VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD RAW DATA
    # --------------------------------------------------------

    df = pd.read_csv(RAW_PATH)

    # Same duplicate removal as preprocessing
    df = df.drop_duplicates()

    # --------------------------------------------------------
    # 2. RECREATE THE EXACT ORIGINAL SPLIT
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

    test_df = test_df.reset_index(drop=True)

    print(f"\nTraining records   : {len(train_df)}")
    print(f"Validation records : {len(validation_df)}")
    print(f"Test records       : {len(test_df)}")

    # --------------------------------------------------------
    # 3. LOAD DNN DETECTOR
    # --------------------------------------------------------

    detector = D1DNNDetector()

    # --------------------------------------------------------
    # 4. RUN DEPLOYMENT DETECTOR
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    print("\nRunning detector on exact test records...")

    for index, row in test_df.iterrows():

        prediction = detector.predict(
            row.to_dict()
        )

        y_true.append(int(row["label"]))
        y_pred.append(int(prediction["class"]))

        if (index + 1) % 1000 == 0:
            print(
                f"Processed: "
                f"{index + 1}/{len(test_df)}"
            )

    # --------------------------------------------------------
    # 5. METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    # --------------------------------------------------------
    # 6. RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEPLOYMENT DETECTOR RESULTS")
    print("=" * 70)

    print(
        f"Accuracy  : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score  : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"FPR       : "
        f"{fpr:.4f} "
        f"({fpr * 100:.2f}%)"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
