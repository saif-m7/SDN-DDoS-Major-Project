import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/raw/dataset1/dataset_sdn.csv"


FEATURES_TO_ANALYZE = [
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
    "port_no",
    "tx_bytes",
    "rx_bytes",
    "tx_kbps",
    "rx_kbps",
    "tot_kbps",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET 1 - LIVE FEATURE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\nSelected feature statistics:")
    print(
        df[FEATURES_TO_ANALYZE]
        .describe()
        .T[
            [
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "mean",
            ]
        ]
    )

    # --------------------------------------------------------
    # Relationship checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DERIVED FEATURE CHECKS")
    print("=" * 70)

    # pktperflow ≈ pktcount / flows?
    calculated_pktperflow = (
        df["pktcount"] /
        df["flows"].replace(0, np.nan)
    )

    pkt_diff = (
        df["pktperflow"] -
        calculated_pktperflow
    ).abs()

    print(
        "\npktperflow vs pktcount / flows:"
    )
    print(
        f"Matching rows: "
        f"{(pkt_diff < 1e-6).sum():,} / {len(df):,}"
    )
    print(
        f"Median absolute difference: "
        f"{pkt_diff.median()}"
    )

    # byteperflow ≈ bytecount / flows?
    calculated_byteperflow = (
        df["bytecount"] /
        df["flows"].replace(0, np.nan)
    )

    byte_diff = (
        df["byteperflow"] -
        calculated_byteperflow
    ).abs()

    print(
        "\nbyteperflow vs bytecount / flows:"
    )
    print(
        f"Matching rows: "
        f"{(byte_diff < 1e-6).sum():,} / {len(df):,}"
    )
    print(
        f"Median absolute difference: "
        f"{byte_diff.median()}"
    )

    # pktrate ≈ pktcount / duration?
    calculated_pktrate = (
        df["pktcount"] /
        df["dur"].replace(0, np.nan)
    )

    rate_diff = (
        df["pktrate"] -
        calculated_pktrate
    ).abs()

    print(
        "\npktrate vs pktcount / dur:"
    )
    print(
        f"Matching rows: "
        f"{(rate_diff < 1e-6).sum():,} / {len(df):,}"
    )
    print(
        f"Median absolute difference: "
        f"{rate_diff.median()}"
    )

    # tot_dur vs dur
    duration_diff = (
        df["tot_dur"] -
        df["dur"]
    ).abs()

    print(
        "\ntot_dur vs dur:"
    )
    print(
        f"Matching rows: "
        f"{(duration_diff < 1e-6).sum():,} / {len(df):,}"
    )
    print(
        f"Median absolute difference: "
        f"{duration_diff.median()}"
    )

    # tot_kbps vs tx_kbps + rx_kbps
    calculated_tot_kbps = (
        df["tx_kbps"] +
        df["rx_kbps"]
    )

    kbps_diff = (
        df["tot_kbps"] -
        calculated_tot_kbps
    ).abs()

    print(
        "\ntot_kbps vs tx_kbps + rx_kbps:"
    )
    print(
        f"Matching rows: "
        f"{(kbps_diff < 1e-6).sum():,} / {len(df):,}"
    )
    print(
        f"Median absolute difference: "
        f"{kbps_diff.median()}"
    )

    # --------------------------------------------------------
    # Pairflow analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIRFLOW ANALYSIS")
    print("=" * 70)

    print("\nPairflow values:")
    print(df["Pairflow"].value_counts())

    print("\nPairflow by label:")
    print(
        pd.crosstab(
            df["Pairflow"],
            df["label"],
            normalize="index"
        )
    )

    # --------------------------------------------------------
    # Packetins analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PACKETINS ANALYSIS")
    print("=" * 70)

    print(
        df["packetins"].describe()
    )

    # --------------------------------------------------------
    # Sample records
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE RECORDS")
    print("=" * 70)

    print(
        df[
            [
                "pktcount",
                "bytecount",
                "dur",
                "tot_dur",
                "flows",
                "packetins",
                "pktperflow",
                "byteperflow",
                "pktrate",
                "Pairflow",
                "tx_kbps",
                "rx_kbps",
                "tot_kbps",
                "label",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()