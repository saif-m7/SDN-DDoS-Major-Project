
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "dataset1"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}

# Columns representing the original 19 model features and label.
# Exclude dt because it is retained for temporal information,
# not used as a DNN input feature.
COMPARE_COLUMNS = [
    "switch", "pktcount", "bytecount", "dur", "dur_nsec",
    "tot_dur", "flows", "packetins", "pktperflow",
    "byteperflow", "pktrate", "Pairflow", "Protocol",
    "port_no", "tx_bytes", "rx_bytes", "tx_kbps",
    "rx_kbps", "tot_kbps", "label",
]


def main():
    datasets = {}

    print("=" * 65)
    print("DATASET 1 SPLIT VALIDATION")
    print("=" * 65)

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} file: {path}")

        df = pd.read_csv(path)

        missing = [c for c in COMPARE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

        datasets[name] = df

        print(f"\n{name.upper()}")
        print(f"Rows: {len(df):,}")
        print(f"Duplicate rows: {df.duplicated().sum():,}")
        print(f"Missing values in model columns: "
              f"{df[COMPARE_COLUMNS].isna().sum().sum():,}")
        print("Label distribution:")
        print(df["label"].value_counts().sort_index().to_dict())

    # Check overlap using the original feature values where possible.
    # The saved CSV features are scaled, so exact rows are compared
    # consistently across all three splits.
    signatures = {
        name: set(
            map(
                tuple,
                df[COMPARE_COLUMNS].itertuples(index=False, name=None)
            )
        )
        for name, df in datasets.items()
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    print("\n" + "=" * 65)
    print("CROSS-SPLIT OVERLAP")
    print("=" * 65)

    for left, right in pairs:
        overlap = signatures[left] & signatures[right]
        print(f"{left} vs {right}: {len(overlap):,} identical feature+label rows")

    print("\nCheck complete. No dataset files were modified.")


if __name__ == "__main__":
    main()
