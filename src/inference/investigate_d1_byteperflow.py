
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")


def compare_values(actual, calculated):
    """Compare actual and calculated values using strict tolerance."""

    valid = actual.notna() & calculated.notna()

    matches = np.isclose(
        actual[valid],
        calculated[valid],
        rtol=0,
        atol=1e-8
    )

    return valid, matches


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "dt",
        "switch",
        "port_no",
        "pktcount",
        "bytecount",
        "flows",
        "pktperflow",
        "byteperflow"
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
    print("DATASET 1 - BYTEPERFLOW INVESTIGATION")
    print("=" * 70)

    print("Total dataset rows:", len(df))

    # --------------------------------------------------
    # 1. Calculate candidate formula
    # --------------------------------------------------

    print("\n1. CANDIDATE FORMULA")
    print("-" * 70)

    print(
        "byteperflow = (bytecount / pktcount) * pktperflow"
    )

    valid_pktcount = (
        df["pktcount"].notna()
        & (df["pktcount"] != 0)
    )

    data = df.loc[valid_pktcount].copy()

    data["calculated_byteperflow"] = (
        data["bytecount"] / data["pktcount"]
    ) * data["pktperflow"]

    data["difference"] = (
        data["byteperflow"]
        - data["calculated_byteperflow"]
    ).abs()

    # --------------------------------------------------
    # 2. Strict comparison
    # --------------------------------------------------

    print("\n2. FORMULA MATCH RESULTS")
    print("-" * 70)

    valid, matches = compare_values(
        data["byteperflow"],
        data["calculated_byteperflow"]
    )

    compared_rows = valid.sum()
    matching_rows = matches.sum()
    mismatching_rows = compared_rows - matching_rows

    match_percentage = (
        matching_rows / compared_rows * 100
        if compared_rows else 0
    )

    mismatch_percentage = (
        mismatching_rows / compared_rows * 100
        if compared_rows else 0
    )

    print("Rows with pktcount != 0:", len(data))
    print("Rows compared:", compared_rows)
    print("Matching rows:", matching_rows)
    print("Mismatching rows:", mismatching_rows)

    print(
        f"Match percentage: {match_percentage:.2f}%"
    )

    print(
        f"Mismatch percentage: {mismatch_percentage:.2f}%"
    )

    # --------------------------------------------------
    # 3. Mismatch analysis
    # --------------------------------------------------

    mismatches = data.loc[
        valid & ~np.isclose(
            data["byteperflow"],
            data["calculated_byteperflow"],
            rtol=0,
            atol=1e-8
        )
    ].copy()

    columns = [
        "dt",
        "switch",
        "port_no",
        "pktcount",
        "bytecount",
        "flows",
        "pktperflow",
        "byteperflow",
        "calculated_byteperflow",
        "difference"
    ]

    print("\n3. FIRST 30 MISMATCHES")
    print("-" * 70)

    if mismatches.empty:
        print("No mismatches found.")
    else:
        print(
            mismatches[columns]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------
    # 4. Difference statistics
    # --------------------------------------------------

    print("\n4. MISMATCH DIFFERENCE STATISTICS")
    print("-" * 70)

    if mismatches.empty:
        print("No mismatch statistics available.")
    else:
        print(
            mismatches["difference"]
            .describe()
            .to_string()
        )

    # --------------------------------------------------
    # 5. Mismatches by switch
    # --------------------------------------------------

    print("\n5. MISMATCHES BY SWITCH")
    print("-" * 70)

    if mismatches.empty:
        print("No mismatches found.")
    else:
        print(
            mismatches.groupby("switch")
            .size()
            .to_string()
        )

    # --------------------------------------------------
    # 6. Zero and negative values
    # --------------------------------------------------

    print("\n6. ZERO AND NEGATIVE VALUE ANALYSIS")
    print("-" * 70)

    print(
        "Rows excluded because pktcount = 0:",
        (~valid_pktcount).sum()
    )

    print(
        "Zero pktperflow:",
        (data["pktperflow"] == 0).sum()
    )

    print(
        "Zero byteperflow:",
        (data["byteperflow"] == 0).sum()
    )

    print(
        "Negative pktperflow:",
        (data["pktperflow"] < 0).sum()
    )

    print(
        "Negative byteperflow:",
        (data["byteperflow"] < 0).sum()
    )

    print(
        "Mismatches with zero pktperflow:",
        (mismatches["pktperflow"] == 0).sum()
    )

    # --------------------------------------------------
    # 7. Mismatch sample
    # --------------------------------------------------

    print("\n7. LAST 20 MISMATCHES")
    print("-" * 70)

    if mismatches.empty:
        print("No mismatches found.")
    else:
        print(
            mismatches[columns]
            .tail(20)
            .to_string(index=False)
        )

    # --------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------

    print("\n8. FINAL SUMMARY")
    print("=" * 70)

    print("Total dataset rows:", len(df))
    print("Rows with nonzero pktcount:", len(data))
    print("Rows compared:", compared_rows)
    print("Matching rows:", matching_rows)
    print("Mismatching rows:", mismatching_rows)

    print(
        f"Formula match accuracy: {match_percentage:.2f}%"
    )

    if match_percentage == 100:
        print(
            "\nConclusion: The candidate formula matches "
            "all compared rows."
        )
    elif match_percentage >= 95:
        print(
            "\nConclusion: The formula matches most rows, "
            "but mismatches require investigation."
        )
    else:
        print(
            "\nConclusion: The formula is not a reliable "
            "universal reconstruction of byteperflow."
        )

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
