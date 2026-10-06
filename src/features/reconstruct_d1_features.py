import pandas as pd
import numpy as np


DATASET_PATH = "data/raw/dataset1/dataset_sdn.csv"


def main():

    print("=" * 70)
    print("D1 FEATURE RECONSTRUCTION")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    # --------------------------------------------------------
    # Calculate possible relationships
    # --------------------------------------------------------

    df["calc_pkt_rate_30"] = df["pktperflow"] / 30

    df["calc_pkt_rate_30_pktcount"] = df["pktcount"] / 30

    df["calc_pktperflow_from_rate"] = df["pktrate"] * 30

    df["calc_byteperflow_from_pktperflow"] = (
        df["byteperflow"] /
        df["pktperflow"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Compare pktrate
    # --------------------------------------------------------

    print("\nPKTRATE relationships")
    print("-" * 70)

    for name in [
        "calc_pkt_rate_30",
        "calc_pkt_rate_30_pktcount",
        "calc_pktperflow_from_rate",
    ]:

        diff = (
            df["pktrate"] -
            df[name]
        ).abs()

        print(
            f"{name:35s} "
            f"median={diff.median():.4f} "
            f"mean={diff.mean():.4f}"
        )

    # --------------------------------------------------------
    # Compare pktperflow with flows
    # --------------------------------------------------------

    print("\nPKTPERFLOW relationships")
    print("-" * 70)

    candidates = {
        "pktcount / flows":
            df["pktcount"] /
            df["flows"].replace(0, np.nan),

        "pktcount / 30":
            df["pktcount"] / 30,

        "pktrate * 30":
            df["pktrate"] * 30,

        "pktrate * flows":
            df["pktrate"] *
            df["flows"],
    }

    for name, candidate in candidates.items():

        diff = (
            df["pktperflow"] -
            candidate
        ).abs()

        print(
            f"{name:35s} "
            f"median={diff.median():.4f} "
            f"mean={diff.mean():.4f}"
        )

    # --------------------------------------------------------
    # Byte-per-packet relationship
    # --------------------------------------------------------

    print("\nBYTEPERFLOW relationships")
    print("-" * 70)

    candidates = {
        "bytecount / pktcount":
            df["bytecount"] /
            df["pktcount"].replace(0, np.nan),

        "byteperflow / pktperflow":
            df["byteperflow"] /
            df["pktperflow"].replace(0, np.nan),

        "bytecount / 30":
            df["bytecount"] / 30,
    }

    for name, candidate in candidates.items():

        diff = (
            df["byteperflow"] -
            candidate
        ).abs()

        print(
            f"{name:35s} "
            f"median={diff.median():.4f} "
            f"mean={diff.mean():.4f}"
        )

    # --------------------------------------------------------
    # Group-level analysis
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
    # Sample rows
    # --------------------------------------------------------

    print("\nSAMPLE CALCULATIONS")
    print("-" * 70)

    columns = [
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

    sample = df[columns].head(20).copy()

    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()