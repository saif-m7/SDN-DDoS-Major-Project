
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "dt",
        "switch",
        "Protocol",
        "pktperflow",
        "pktrate"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("=" * 70)
    print("DATASET 1 - PKTRATE INVESTIGATION")
    print("=" * 70)

    print("Total records:", len(df))

    # --------------------------------------------------
    # 1. Overall statistics
    # --------------------------------------------------

    print("\n1. OVERALL STATISTICS")
    print("-" * 70)

    print("\nPKTRATE:")
    print(df["pktrate"].describe().to_string())

    print("\nPKTPERFLOW:")
    print(df["pktperflow"].describe().to_string())

    # --------------------------------------------------
    # 2. Candidate formula
    # --------------------------------------------------

    print("\n2. CANDIDATE FORMULA")
    print("-" * 70)

    print("pktrate = floor(pktperflow / 30)")

    calculated = np.floor(
        df["pktperflow"] / 30
    )

    valid = (
        df["pktrate"].notna()
        & calculated.notna()
    )

    matches = np.isclose(
        df.loc[valid, "pktrate"],
        calculated[valid],
        rtol=0,
        atol=1e-8
    )

    total = valid.sum()
    matching = matches.sum()
    mismatching = total - matching

    print("\nRows compared:", total)
    print("Matching rows:", matching)
    print("Mismatching rows:", mismatching)

    percentage = (
        matching / total * 100
        if total else 0
    )

    print(f"Match percentage: {percentage:.4f}%")

    # --------------------------------------------------
    # 3. Mismatch analysis
    # --------------------------------------------------

    print("\n3. MISMATCH ANALYSIS")
    print("-" * 70)

    comparison = df.loc[valid].copy()

    comparison["calculated_pktrate"] = calculated[valid]

    comparison["difference"] = (
        comparison["pktrate"]
        - comparison["calculated_pktrate"]
    ).abs()

    mismatches = comparison.loc[
        ~np.isclose(
            comparison["pktrate"],
            comparison["calculated_pktrate"],
            rtol=0,
            atol=1e-8
        )
    ]

    if mismatches.empty:
        print("No mismatches found.")
    else:
        columns = [
            "dt",
            "switch",
            "Protocol",
            "pktperflow",
            "pktrate",
            "calculated_pktrate",
            "difference"
        ]

        print("\nFirst 30 mismatches:")
        print(
            mismatches[columns]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------
    # 4. Negative and zero values
    # --------------------------------------------------

    print("\n4. NEGATIVE AND ZERO VALUE ANALYSIS")
    print("-" * 70)

    print(
        "Negative pktperflow:",
        (df["pktperflow"] < 0).sum()
    )

    print(
        "Negative pktrate:",
        (df["pktrate"] < 0).sum()
    )

    print(
        "Zero pktperflow:",
        (df["pktperflow"] == 0).sum()
    )

    print(
        "Zero pktrate:",
        (df["pktrate"] == 0).sum()
    )

    # --------------------------------------------------
    # 5. Protocol-wise validation
    # --------------------------------------------------

    print("\n5. MATCHES BY PROTOCOL")
    print("-" * 70)

    for protocol, group in comparison.groupby("Protocol"):

        group_matches = np.isclose(
            group["pktrate"],
            group["calculated_pktrate"],
            rtol=0,
            atol=1e-8
        )

        count = len(group)
        correct = group_matches.sum()

        accuracy = (
            correct / count * 100
            if count else 0
        )

        print(
            f"{protocol}: "
            f"{correct}/{count} "
            f"({accuracy:.2f}%)"
        )

    # --------------------------------------------------
    # 6. Switch-wise validation
    # --------------------------------------------------

    print("\n6. MATCHES BY SWITCH")
    print("-" * 70)

    for switch, group in comparison.groupby("switch"):

        group_matches = np.isclose(
            group["pktrate"],
            group["calculated_pktrate"],
            rtol=0,
            atol=1e-8
        )

        count = len(group)
        correct = group_matches.sum()

        accuracy = (
            correct / count * 100
            if count else 0
        )

        print(
            f"Switch {switch}: "
            f"{correct}/{count} "
            f"({accuracy:.2f}%)"
        )

    # --------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------

    print("\n7. FINAL SUMMARY")
    print("=" * 70)

    print("Total records:", len(df))
    print("Rows compared:", total)
    print("Matching rows:", matching)
    print("Mismatching rows:", mismatching)
    print(f"Match percentage: {percentage:.4f}%")

    if percentage == 100:
        print(
            "\nConclusion: The candidate formula matches "
            "every compared record."
        )
    else:
        print(
            "\nConclusion: The candidate formula does not "
            "match every record. Further investigation is needed."
        )

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
