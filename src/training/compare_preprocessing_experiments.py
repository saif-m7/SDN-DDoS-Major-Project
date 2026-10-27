
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = {
    "StandardScaler": (
        PROJECT_ROOT / "results" / "experiments"
        / "dataset1_epoch_comparison"
    ),
    "MinMaxScaler": (
        PROJECT_ROOT / "results" / "experiments"
        / "dataset1_minmax_dnn"
    ),
    "RobustScaler": (
        PROJECT_ROOT / "results" / "experiments"
        / "dataset1_robust_dnn"
    ),
    "Shifted Log Transformation": (
        PROJECT_ROOT / "results" / "experiments"
        / "dataset1_log_dnn"
    ),
}

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "experiments"
    / "dataset1_preprocessing_comparison"
)


# ============================================================
# FIND RESULTS CSV
# ============================================================

def find_results_csv(folder):
    if not folder.exists():
        raise FileNotFoundError(
            f"Experiment folder not found:\n{folder}"
        )

    candidates = sorted(folder.rglob("*comparison*.csv"))

    if not candidates:
        candidates = sorted(folder.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No result CSV found in:\n{folder}\n"
            "Make sure the experiment has finished and saved its results."
        )

    if len(candidates) > 1:
        print(f"\nMultiple CSV files found in {folder.name}:")
        for candidate in candidates:
            print(f"  - {candidate.name}")

        # Prefer the CSV that explicitly identifies the epoch comparison.
        preferred = [
            path for path in candidates
            if "epoch_comparison" in path.name.lower()
        ]

        if preferred:
            candidates = preferred

        print(f"Using: {candidates[0].name}")

    return candidates[0]


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):
    df = df.copy()

    df.columns = [
        str(column).strip().lower()
        .replace("-", "_")
        .replace(" ", "_")
        for column in df.columns
    ]

    # Map different experiment-script column names to common names.
    aliases = {
        # Epochs
        "epochs_requested": "epochs",
        "epochs_completed": "epochs",
        "epoch": "epochs",

        # Training accuracy
        "training_accuracy_last_epoch": "train_accuracy",
        "final_train_accuracy": "train_accuracy",
        "training_accuracy": "train_accuracy",
        "train_acc": "train_accuracy",

        # Validation accuracy
        "validation_accuracy_last_epoch": "validation_accuracy",
        "final_validation_accuracy": "validation_accuracy",
        "val_accuracy": "validation_accuracy",
        "val_acc": "validation_accuracy",

        # Test accuracy
        "test_acc": "test_accuracy",

        # Precision, recall, F1
        "test_precision": "precision",
        "precision_score": "precision",
        "test_recall": "recall",
        "recall_score": "recall",
        "test_f1_score": "f1_score",
        "f1": "f1_score",
        "f1score": "f1_score",

        # False-positive rate
        "test_false_positive_rate": "false_positive_rate",
        "fpr": "false_positive_rate",

        # Timing
        "training_time": "training_time_seconds",
        "training_time_sec": "training_time_seconds",
        "test_inference_time": "test_inference_time_seconds",
    }

    for old_name, new_name in aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    return df


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

def convert_numeric_columns(df):
    df = df.copy()

    numeric_columns = [
        "epochs",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "training_time_seconds",
        "test_inference_time_seconds",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "batch_size",
        "learning_rate",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# LOAD ONE EXPERIMENT
# ============================================================

def load_one_experiment(preprocessing, folder):
    print("\n" + "=" * 70)
    print(f"Loading: {preprocessing}")
    print(f"Folder: {folder}")

    csv_path = find_results_csv(folder)

    df = pd.read_csv(csv_path)
    df = standardize_columns(df)
    df = convert_numeric_columns(df)

    required_columns = ["epochs", "validation_accuracy"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in:\n"
            f"{csv_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    if df["epochs"].isna().all():
        raise ValueError(
            f"No valid epoch values found in:\n{csv_path}"
        )

    if df["validation_accuracy"].isna().all():
        raise ValueError(
            f"No valid validation accuracy values found in:\n{csv_path}"
        )

    df["preprocessing"] = preprocessing
    df["source_csv"] = str(
        csv_path.relative_to(PROJECT_ROOT)
    )

    print(f"Loaded file: {csv_path.name}")
    print(f"Experiment rows: {len(df)}")
    print(f"Available columns: {list(df.columns)}")

    return df


# ============================================================
# LOAD ALL FOUR EXPERIMENTS
# ============================================================

def load_all_experiments():
    all_results = []

    for preprocessing, folder in EXPERIMENTS.items():
        df = load_one_experiment(preprocessing, folder)
        all_results.append(df)

    return pd.concat(
        all_results,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# CREATE BEST-EPOCH SUMMARY
# ============================================================

def create_summary(combined):
    summary_rows = []

    for preprocessing in EXPERIMENTS:
        group = combined[
            combined["preprocessing"] == preprocessing
        ].copy()

        valid_group = group.dropna(
            subset=["epochs", "validation_accuracy"]
        )

        if valid_group.empty:
            print(
                f"Warning: no valid validation results for "
                f"{preprocessing}"
            )
            continue

        # Select the epoch configuration with the highest validation
        # accuracy. Test metrics are reported for that same row.
        best = valid_group.loc[
            valid_group["validation_accuracy"].idxmax()
        ]

        row = {
            "preprocessing": preprocessing,
            "selected_epochs": int(best["epochs"]),
            "validation_accuracy": best["validation_accuracy"],
        }

        metric_columns = [
            "train_accuracy",
            "test_accuracy",
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
            "training_time_seconds",
            "test_inference_time_seconds",
        ]

        for column in metric_columns:
            if column in best.index:
                row[column] = best[column]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DATASET 1 - FOUR PREPROCESSING EXPERIMENT COMPARISON")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results from all four preprocessing experiments.
    combined = load_all_experiments()

    # Sort results into a consistent preprocessing and epoch order.
    order = {
        "StandardScaler": 0,
        "MinMaxScaler": 1,
        "RobustScaler": 2,
        "Shifted Log Transformation": 3,
    }

    combined["_order"] = combined["preprocessing"].map(order)

    combined = combined.sort_values(
        ["_order", "epochs"]
    ).drop(columns="_order")

    # Save all epoch results.
    combined_csv = (
        OUTPUT_DIR / "all_preprocessing_results.csv"
    )

    combined.to_csv(
        combined_csv,
        index=False,
    )

    # Create the best-epoch summary.
    summary = create_summary(combined)

    summary_csv = (
        OUTPUT_DIR / "best_preprocessing_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    # Print available metrics.
    display_columns = [
        "preprocessing",
        "selected_epochs",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "training_time_seconds",
        "test_inference_time_seconds",
    ]

    available_columns = [
        column for column in display_columns
        if column in summary.columns
    ]

    print("\n" + "=" * 110)
    print("BEST EPOCH CONFIGURATION FOR EACH PREPROCESSING METHOD")
    print("=" * 110)

    if summary.empty:
        print("No valid summary could be generated.")
    else:
        print(
            summary[available_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)
    print(f"All epoch results: {combined_csv}")
    print(f"Best-epoch summary: {summary_csv}")

    print(
        "\nSelection rule: highest validation accuracy per method. "
        "Test metrics correspond to that selected epoch."
    )


if __name__ == "__main__":
    main()
