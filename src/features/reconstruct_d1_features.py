
import pandas as pd
import numpy as np


DATASET_PATH = "data/raw/dataset1/dataset_sdn.csv"


def main():

    print("=" * 70)
    print("D1 FEATURE RECONSTRUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "dt", "switch", "port_no",
        "pktcount", "bytecount", "flows",
        "pktperflow", "byteperflow", "pktrate"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. POSSIBLE FEATURE RELATIONSHIPS
    # --------------------------------------------------------

    df["calc_pkt_rate_30"] = df["pktperflow"] / 30
    df["calc_pktcount_rate_30"] = df["pktcount"] / 30
    df["calc_pktperflow_from_rate"] = df["pktrate"] * 30

    # --------------------------------------------------------
    # 3. PKTRATE ANALYSIS
    # --------------------------------------------------------

    print("\nPKTRATE RELATIONSHIPS")
    print("-" * 70)

    candidates = {
        "pktperflow / 30": df["pktperflow"] / 30,
        "pktcount / 30": df["pktcount"] / 30,
        "pktrate * 30": df["pktrate"] * 30,
    }

    for name, candidate in candidates.items():

        diff = (df["pktrate"] - candidate).abs()

        print(
            f"{name:30s} "
            f"median={diff.median():.4f} "
            f"mean={diff.mean():.4f}"
        )

    # --------------------------------------------------------
    # 4. PKTPERFLOW ANALYSIS
    # --------------------------------------------------------

    print("\nPKTPERFLOW RELATIONSHIPS")
    print("-" * 70)

    candidates = {
        "pktcount / flows":
            df["pktcount"] / df["flows"].replace(0, np.nan),

        "pktcount / 30":
            df["pktcount"] / 30,

        "pktrate * 30":
            df["pktrate"] * 30,

        "pktrate * flows":
            df["pktrate"] * df["flows"],
    }

    for name, candidate in candidates.items():

        diff = (df["pktperflow"] - candidate).abs()

        print(
            f"{name:30s} "
            f"median={diff.median():.4f} "
            f"mean={diff.mean():.4f}"
        )

    # --------------------------------------------------------
    # 5. BYTEPERFLOW ANALYSIS
    # --------------------------------------------------------

    print("\nBYTEPERFLOW RELATIONSHIPS")
    print("-" * 70)

    candidates = {
        "bytecount / pktcount":
            df["bytecount"] / df["pktcount"].replace(0, np.nan),

        "bytecount / flows":
            df["bytecount"] / df["flows"].replace(0, np.nan),

        "bytecount / 30":
            df["bytecount"] / 30,

        "(bytecount / pktcount) * pktperflow":
            (
                df["bytecount"] /
                df["pktcount"].replace(0, np.nan)
            ) * df["pktperflow"],
    }

    for name, candidate in candidates.items():

        diff = (df["byteperflow"] - candidate).abs()

        print(
            f"{name:45s} "
            f"median={diff.median():.4f} "
            f"mean={diff.mean():.4f}"
        )

    # --------------------------------------------------------
    # 6. EXACT FORMULA VERIFICATION
    # --------------------------------------------------------

    print("\nEXACT FORMULA VERIFICATION")
    print("-" * 70)

    # PKTRATE

    pkt_rate_calc = df["pktperflow"] / 30

    valid_rate = (
        df["pktrate"].notna() &
        pkt_rate_calc.notna()
    )

    actual_rate = df.loc[valid_rate, "pktrate"]
    calculated_rate = pkt_rate_calc.loc[valid_rate]

    print("\nPKTRATE:")

    print(
        "Exact matches:",
        np.isclose(actual_rate, calculated_rate).sum()
    )

    print(
        "Rounded matches:",
        np.isclose(
            actual_rate,
            np.round(calculated_rate)
        ).sum()
    )

    print(
        "Floor matches:",
        np.isclose(
            actual_rate,
            np.floor(calculated_rate)
        ).sum()
    )

    print("Valid rows:", valid_rate.sum())

    # BYTEPERFLOW

    valid_bytes = (
        (df["pktcount"] != 0) &
        df["byteperflow"].notna()
    )

    byte_calc = (
        df.loc[valid_bytes, "bytecount"] /
        df.loc[valid_bytes, "pktcount"]
    ) * df.loc[valid_bytes, "pktperflow"]

    actual_bytes = df.loc[valid_bytes, "byteperflow"]

    print("\nBYTEPERFLOW:")

    print(
        "Exact matches:",
        np.isclose(actual_bytes, byte_calc).sum()
    )

    print(
        "Rounded matches:",
        np.isclose(
            actual_bytes,
            np.round(byte_calc)
        ).sum()
    )

    print(
        "Floor matches:",
        np.isclose(
            actual_bytes,
            np.floor(byte_calc)
        ).sum()
    )

    print("Valid rows:", valid_bytes.sum())

    # --------------------------------------------------------
    # 7. GROUP-LEVEL ANALYSIS
    # --------------------------------------------------------

    print("\nGROUP ANALYSIS")
    print("-" * 70)

    group_columns = [
        ["dt"],
        ["dt", "switch"],
        ["dt", "switch", "port_no"],
    ]

    for columns in group_columns:

        grouped = (
            df.groupby(columns)
            .agg(
                pktcount=("pktcount", "sum"),
                bytecount=("bytecount", "sum"),
                pktperflow=("pktperflow", "mean"),
                byteperflow=("byteperflow", "mean"),
                pktrate=("pktrate", "mean"),
                flows=("flows", "mean"),
            )
            .reset_index()
        )

        print(f"\nGrouping: {columns}")
        print(grouped.head(10).to_string(index=False))

    # --------------------------------------------------------
    # 8. SAMPLE CALCULATIONS
    # --------------------------------------------------------

    print("\nSAMPLE CALCULATIONS")
    print("-" * 70)

    sample_columns = [
        "dt",
        "switch",
        "port_no",
        "pktcount",
        "flows",
        "pktperflow",
        "pktrate",
        "bytecount",
        "byteperflow",
    ]

    print(
        df[sample_columns]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 9. COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("D1 FEATURE RECONSTRUCTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
