import pandas as pd
import numpy as np


DATASET_PATH = "data/raw/dataset1/dataset_sdn.csv"


def main():

    print("=" * 70)
    print("DATASET 1 - FEATURE RELATIONSHIP INVESTIGATION")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    # Sort according to dataset time
    df = df.sort_values("dt").reset_index(drop=True)

    # --------------------------------------------------------
    # 1. Check dt
    # --------------------------------------------------------

    print("\nDT information:")
    print(df["dt"].describe())

    # --------------------------------------------------------
    # 2. Check whether features are related to time deltas
    # --------------------------------------------------------

    df["dt_delta"] = df["dt"].diff()

    print("\nDT delta distribution:")
    print(df["dt_delta"].describe())

    # --------------------------------------------------------
    # 3. Check pktrate relationships
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PKTRATE INVESTIGATION")
    print("=" * 70)

    calculations = {
        "pktcount / dur":
            df["pktcount"] /
            df["dur"].replace(0, np.nan),

        "pktcount / (dur + dur_nsec/1e9)":
            df["pktcount"] /
            (
                df["dur"] +
                df["dur_nsec"] / 1e9
            ).replace(0, np.nan),

        "pktcount / dt_delta":
            df["pktcount"] /
            df["dt_delta"].replace(0, np.nan),

        "pktperflow / dur":
            df["pktperflow"] /
            df["dur"].replace(0, np.nan),

        "pktperflow / dt_delta":
            df["pktperflow"] /
            df["dt_delta"].replace(0, np.nan),
    }

    for name, values in calculations.items():

        diff = (
            df["pktrate"] - values
        ).abs()

        print(f"\n{name}")
        print(
            f"Median absolute difference: "
            f"{diff.median()}"
        )
        print(
            f"Mean absolute difference: "
            f"{diff.mean()}"
        )

    # --------------------------------------------------------
    # 4. Check byteperflow relationships
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BYTEPERFLOW INVESTIGATION")
    print("=" * 70)

    calculations = {
        "bytecount / flows":
            df["bytecount"] /
            df["flows"].replace(0, np.nan),

        "bytecount / pktcount":
            df["bytecount"] /
            df["pktcount"].replace(0, np.nan),

        "byteperflow / pktperflow":
            df["byteperflow"] /
            df["pktperflow"].replace(0, np.nan),
    }

    for name, values in calculations.items():

        diff = (
            df["byteperflow"] - values
        ).abs()

        print(f"\n{name}")
        print(
            f"Median absolute difference: "
            f"{diff.median()}"
        )
        print(
            f"Mean absolute difference: "
            f"{diff.mean()}"
        )

    # --------------------------------------------------------
    # 5. Pairflow by common flow identifiers
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIRFLOW RELATIONSHIP")
    print("=" * 70)

    # Check Pairflow by switch/port/protocol
    for columns in [
        ["switch"],
        ["switch", "port_no"],
        ["switch", "Protocol"],
        ["switch", "port_no", "Protocol"],
    ]:

        grouped = (
            df.groupby(columns)["Pairflow"]
            .agg(["mean", "count", "nunique"])
            .reset_index()
        )

        print(
            f"\nGrouping by: {columns}"
        )

        print(grouped.head(15).to_string(index=False))

    # --------------------------------------------------------
    # 6. Pairflow transition behavior
    # --------------------------------------------------------

    print("\nPairflow transitions:")

    pairflow_previous = df["Pairflow"].shift(1)

    transitions = pd.crosstab(
        pairflow_previous,
        df["Pairflow"]
    )

    print(transitions)

    # --------------------------------------------------------
    # 7. Feature correlations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRELATION WITH TARGET")
    print("=" * 70)

    features = [
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
        "tx_bytes",
        "rx_bytes",
        "tx_kbps",
        "rx_kbps",
        "tot_kbps",
    ]

    correlations = (
        df[features + ["label"]]
        .corr(numeric_only=True)["label"]
        .drop("label")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(correlations)

    # --------------------------------------------------------
    # 8. Show consecutive examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONSECUTIVE DATASET RECORDS")
    print("=" * 70)

    columns = [
        "dt",
        "switch",
        "port_no",
        "pktcount",
        "bytecount",
        "dur",
        "dur_nsec",
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

    print(
        df[columns]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()