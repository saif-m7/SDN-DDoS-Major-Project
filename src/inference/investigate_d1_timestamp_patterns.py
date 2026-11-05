
import pandas as pd
import numpy as np

DATASET_PATH = "data/raw/dataset1/dataset_sdn.csv"


def main():

    df = pd.read_csv(DATASET_PATH)

    print("=" * 75)
    print("TIMESTAMP AND TRAFFIC PATTERN INVESTIGATION")
    print("=" * 75)

    # 1. Inspect negative pktperflow rows
    print("\n1. NEGATIVE PKTPERFLOW RECORDS")
    print("-" * 75)

    negative = df[df["pktperflow"] < 0]

    print("Negative rows:", len(negative))

    columns = [
        "dt", "switch", "port_no", "Protocol",
        "pktcount", "bytecount", "dur",
        "tot_dur", "flows", "pktperflow",
        "byteperflow", "pktrate"
    ]

    print(negative[columns].head(30).to_string(index=False))

    # 2. Inspect timestamp 2740 and 2741
    print("\n2. DETAILED TIMESTAMP INSPECTION")
    print("-" * 75)

    for timestamp in [2740, 2741, 2488, 2518]:

        subset = df[df["dt"] == timestamp]

        print(f"\nTimestamp: {timestamp}")
        print("Rows:", len(subset))

        print(
            subset[columns]
            .sort_values(["switch", "port_no"])
            .head(35)
            .to_string(index=False)
        )

    # 3. Check whether the same packet/byte counters
    # have different pktperflow values.
    print("\n3. SAME COUNTERS, DIFFERENT PKTPERFLOW")
    print("-" * 75)

    grouped = (
        df.groupby(["dt", "pktcount", "bytecount"])
        .agg(
            rows=("pktperflow", "size"),
            unique_pktperflow=("pktperflow", "nunique"),
            min_pktperflow=("pktperflow", "min"),
            max_pktperflow=("pktperflow", "max"),
            unique_switches=("switch", "nunique"),
        )
        .reset_index()
    )

    different = grouped[
        grouped["unique_pktperflow"] > 1
    ]

    print("Groups with different pktperflow:",
          len(different))

    print(
        different.head(30).to_string(index=False)
    )

    # 4. Inspect dt progression
    print("\n4. TIMESTAMP INTERVALS")
    print("-" * 75)

    timestamps = np.sort(df["dt"].unique())

    intervals = np.diff(timestamps)

    print("Unique timestamps:", len(timestamps))
    print("Interval distribution:")
    print(pd.Series(intervals).value_counts().head(15))

    # 5. Negative values by protocol and switch
    print("\n5. NEGATIVE VALUES BY PROTOCOL")
    print("-" * 75)

    print(
        negative.groupby("Protocol")
        .size()
        .to_string()
    )

    print("\nNegative values by switch:")
    print(
        negative.groupby("switch")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
