
import pandas as pd
import numpy as np

PATH = "data/raw/dataset1/dataset_sdn.csv"


def main():

    df = pd.read_csv(PATH)

    expected = np.round(
        df["dur"] + df["dur_nsec"] / 1e9
    ) * 1e9

    df["expected_tot_dur"] = expected
    df["difference"] = (
        df["tot_dur"] - expected
    ) / 1e9

    mismatches = df[
        ~np.isclose(
            df["tot_dur"],
            expected,
            rtol=0,
            atol=0
        )
    ].copy()

    print("=" * 70)
    print("DURATION MISMATCH INVESTIGATION")
    print("=" * 70)

    print("\nTotal rows:", len(df))
    print("Mismatches:", len(mismatches))

    print("\nDIFFERENCE DISTRIBUTION")
    print("-" * 70)

    print(
        mismatches["difference"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMISMATCHES BY PROTOCOL")
    print("-" * 70)

    print(
        mismatches.groupby("Protocol")
        .size()
        .to_string()
    )

    print("\nMISMATCHES BY SWITCH")
    print("-" * 70)

    print(
        mismatches.groupby("switch")
        .size()
        .to_string()
    )

    print("\nFIRST 30 MISMATCHES")
    print("-" * 70)

    columns = [
        "dt", "switch", "port_no", "Protocol",
        "dur", "dur_nsec", "tot_dur",
        "expected_tot_dur", "difference"
    ]

    print(
        mismatches[columns]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
