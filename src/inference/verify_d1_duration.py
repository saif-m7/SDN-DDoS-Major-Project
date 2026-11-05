
import pandas as pd
import numpy as np

PATH = "data/raw/dataset1/dataset_sdn.csv"


def check_formula(name, actual, calculated):

    valid = actual.notna() & calculated.notna()

    exact = np.isclose(
        actual[valid],
        calculated[valid],
        rtol=0,
        atol=0
    )

    diff = (actual[valid] - calculated[valid]).abs()

    print(f"\n{name}")
    print("-" * 65)
    print("Exact matches:", exact.sum())
    print("Match percentage:",
          round(exact.mean() * 100, 2), "%")
    print("Median difference:", diff.median())
    print("Maximum difference:", diff.max())


def main():

    df = pd.read_csv(PATH)

    actual = df["tot_dur"]

    print("=" * 70)
    print("DURATION FORMULA RECONSTRUCTION")
    print("=" * 70)

    candidates = {
        "dur * 1e9":
            df["dur"] * 1e9,

        "(dur * 1e9) + dur_nsec":
            df["dur"] * 1e9 + df["dur_nsec"],

        "(dur + 1) * 1e9":
            (df["dur"] + 1) * 1e9,

        "round(dur_nsec / 1e9 + dur) * 1e9":
            np.round(
                df["dur"] + df["dur_nsec"] / 1e9
            ) * 1e9,

        "ceil(dur_nsec / 1e9 + dur) * 1e9":
            np.ceil(
                df["dur"] + df["dur_nsec"] / 1e9
            ) * 1e9,

        "floor(dur_nsec / 1e9 + dur) * 1e9":
            np.floor(
                df["dur"] + df["dur_nsec"] / 1e9
            ) * 1e9,
    }

    for name, calculated in candidates.items():
        check_formula(name, actual, calculated)

    print("\nDURATION FIELD RELATIONSHIP")
    print("-" * 70)

    df["tot_dur_seconds"] = actual / 1e9

    print(
        df[["dur", "dur_nsec", "tot_dur_seconds"]]
        .head(30)
        .to_string(index=False)
    )

    print("\nDIFFERENCE BETWEEN TOT_DUR AND DUR")
    print("-" * 70)

    print(
        (df["tot_dur_seconds"] - df["dur"])
        .describe()
    )


if __name__ == "__main__":
    main()
