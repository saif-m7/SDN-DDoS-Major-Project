import json
from pathlib import Path

# ============================================================
# MODEL COMPARISON
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(filename):
    path = METRICS_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# GET FIRST AVAILABLE VALUE
# ============================================================

def get_first(data, keys):
    """
    Return the first existing, non-None value
    from the supplied list of possible JSON keys.
    """

    for key in keys:
        value = data.get(key)

        if value is not None:
            return value

    return None


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_metrics(dataset, model, filename):

    data = load_json(filename)

    accuracy = get_first(
        data,
        [
            "accuracy"
        ]
    )

    precision = get_first(
        data,
        [
            "precision",
            "precision_weighted",
            "weighted_precision"
        ]
    )

    recall = get_first(
        data,
        [
            "recall",
            "recall_weighted",
            "weighted_recall"
        ]
    )

    f1_score = get_first(
        data,
        [
            "f1_score",
            "f1_score_weighted",
            "weighted_f1_score"
        ]
    )

    false_positive_rate = get_first(
        data,
        [
            "false_positive_rate",
            "false_positive_rate_macro",
            "macro_false_positive_rate",
            "macro_fpr"
        ]
    )

    # Make sure the metric file contains everything required
    required_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "FPR": false_positive_rate
    }

    missing = [
        name
        for name, value in required_metrics.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            f"Missing metrics in {filename}: {', '.join(missing)}"
        )

    return {
        "Dataset": dataset,
        "Model": model,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "FPR": false_positive_rate,
    }


# ============================================================
# DATASET 1
# ============================================================

dataset1_results = [

    extract_metrics(
        "Dataset 1",
        "DNN",
        "dnn_dataset1_metrics.json"
    ),

    extract_metrics(
        "Dataset 1",
        "CNN",
        "cnn_dataset1_metrics.json"
    ),

    extract_metrics(
        "Dataset 1",
        "LSTM",
        "lstm_dataset1_metrics.json"
    ),

    extract_metrics(
        "Dataset 1",
        "TabNet",
        "tabnet_dataset1_metrics.json"
    ),
]


# ============================================================
# DATASET 2
# ============================================================

dataset2_results = [

    extract_metrics(
        "Dataset 2",
        "DNN",
        "dnn_dataset2_metrics.json"
    ),

    extract_metrics(
        "Dataset 2",
        "CNN",
        "cnn_dataset2_metrics.json"
    ),

    extract_metrics(
        "Dataset 2",
        "TabNet",
        "tabnet_dataset2_metrics.json"
    ),
]


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = dataset1_results + dataset2_results


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)


for dataset in ["Dataset 1", "Dataset 2"]:

    print(f"\n{dataset}")
    print("-" * 80)

    print(
        f"{'Model':<10}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'FPR':>12}"
    )

    print("-" * 80)

    for result in all_results:

        if result["Dataset"] != dataset:
            continue

        print(
            f"{result['Model']:<10}"
            f"{result['Accuracy'] * 100:>11.2f}%"
            f"{result['Precision'] * 100:>11.2f}%"
            f"{result['Recall'] * 100:>11.2f}%"
            f"{result['F1-score'] * 100:>11.2f}%"
            f"{result['FPR'] * 100:>11.2f}%"
        )


# ============================================================
# SAVE JSON REPORT
# ============================================================

output_file = REPORTS_DIR / "model_comparison.json"

comparison_report = {
    "dataset1": dataset1_results,
    "dataset2": dataset2_results
}


with open(output_file, "w", encoding="utf-8") as file:

    json.dump(
        comparison_report,
        file,
        indent=4
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETED")
print("=" * 80)

print("\nComparison report saved to:")
print(output_file)

print("=" * 80)