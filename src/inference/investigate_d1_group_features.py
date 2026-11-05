
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_distribution(series, name):
    print(f"\n{name} distribution:")
    print(series.value_counts(dropna=False).sort_index().to_string())


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "dt", "switch", "src", "dst", "Protocol",
        "port_no", "flows", "packetins", "Pairflow",
        "pktcount", "bytecount", "label"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in [
        "dt", "switch", "port_no", "flows",
        "packetins", "Pairflow", "pktcount",
        "bytecount", "label"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(
        ["switch", "dt"]
    ).reset_index(drop=True)

    section("DATASET OVERVIEW")

    print("Records:", len(df))
    print("Unique timestamps:", df["dt"].nunique())
    print("Unique switches:", df["switch"].nunique())
    print("Unique source-destination-protocol tuples:",
          df[["src", "dst", "Protocol"]].drop_duplicates().shape[0])

    # --------------------------------------------------
    # 1. FLOWS FEATURE
    # --------------------------------------------------

    section("1. FLOWS FEATURE INVESTIGATION")

    print("\nOverall distribution:")
    print(df["flows"].describe().to_string())

    show_distribution(df["flows"], "flows")

    # Check whether flows is constant within timestamp + switch.
    group_columns = ["dt", "switch"]

    grouped = df.groupby(
        group_columns,
        dropna=False
    )

    flow_summary = grouped.agg(
        record_count=("flows", "size"),
        flows_min=("flows", "min"),
        flows_max=("flows", "max"),
        flows_mean=("flows", "mean"),
        unique_flows_values=("flows", "nunique"),
        unique_src_dst_protocol=(
            "src",
            "nunique"
        )
    ).reset_index()

    consistent = (
        flow_summary["unique_flows_values"] == 1
    ).sum()

    print("\nTimestamp + switch groups:")
    print("Total groups:", len(flow_summary))
    print("Groups with one flows value:", consistent)
    print(
        "Consistency percentage:",
        round(consistent / len(flow_summary) * 100, 2),
        "%"
    )

    print("\nExample timestamp + switch groups:")
    print(flow_summary.head(20).to_string(index=False))

    # Compare flows with distinct source-destination-protocol tuples.
    tuple_counts = (
        df.groupby(["dt", "switch"], dropna=False)
        .apply(
            lambda g: g[
                ["src", "dst", "Protocol"]
            ].drop_duplicates().shape[0],
            include_groups=False
        )
        .rename("unique_src_dst_protocol_count")
        .reset_index()
    )

    flow_comparison = flow_summary.merge(
        tuple_counts,
        on=["dt", "switch"],
        how="left"
    )

    flow_comparison["difference"] = (
        flow_comparison["flows_min"]
        - flow_comparison["unique_src_dst_protocol_count"]
    )

    exact_matches = (
        flow_comparison["difference"] == 0
    ).sum()

    print("\nFlows vs unique source-destination-protocol tuples:")
    print("Comparable groups:", len(flow_comparison))
    print("Exact matches:", exact_matches)
    print(
        "Match percentage:",
        round(exact_matches / len(flow_comparison) * 100, 2),
        "%"
    )

    print("\nLargest differences:")
    print(
        flow_comparison.assign(
            abs_difference=lambda x: x["difference"].abs()
        )
        .sort_values("abs_difference", ascending=False)
        .head(20)
        .drop(columns="abs_difference")
        .to_string(index=False)
    )

    # Check flows over time for each switch.
    flow_time = (
        df.groupby(["switch", "dt"], as_index=False)
        .agg(flows=("flows", "first"))
        .sort_values(["switch", "dt"])
    )

    flow_time["previous_flows"] = (
        flow_time.groupby("switch")["flows"].shift(1)
    )

    flow_time["flows_change"] = (
        flow_time["flows"] - flow_time["previous_flows"]
    )

    print("\nFlows changes over time:")
    print(
        flow_time["flows_change"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------
    # 2. PACKETINS FEATURE
    # --------------------------------------------------

    section("2. PACKETINS FEATURE INVESTIGATION")

    print("\nOverall statistics:")
    print(df["packetins"].describe().to_string())

    show_distribution(
        df["packetins"],
        "packetins"
    )

    packetin_groups = (
        df.groupby(["dt", "switch"], dropna=False)
        .agg(
            packetins_min=("packetins", "min"),
            packetins_max=("packetins", "max"),
            packetins_mean=("packetins", "mean"),
            unique_packetins=("packetins", "nunique"),
            records=("packetins", "size")
        )
        .reset_index()
    )

    consistent_packetins = (
        packetin_groups["unique_packetins"] == 1
    ).sum()

    print("\nPacketins consistency within timestamp + switch:")
    print("Total groups:", len(packetin_groups))
    print("Consistent groups:", consistent_packetins)
    print(
        "Consistency percentage:",
        round(
            consistent_packetins / len(packetin_groups) * 100,
            2
        ),
        "%"
    )

    # One packetins value per switch and timestamp.
    packetin_time = (
        packetin_groups[
            ["dt", "switch", "packetins_mean"]
        ]
        .sort_values(["switch", "dt"])
        .copy()
    )

    packetin_time["previous_packetins"] = (
        packetin_time.groupby("switch")[
            "packetins_mean"
        ].shift(1)
    )

    packetin_time["packetins_change"] = (
        packetin_time["packetins_mean"]
        - packetin_time["previous_packetins"]
    )

    valid_changes = packetin_time[
        "packetins_change"
    ].dropna()

    print("\nPacketins changes between consecutive timestamps:")
    print("Positive changes:", (valid_changes > 0).sum())
    print("Negative changes:", (valid_changes < 0).sum())
    print("Unchanged:", (valid_changes == 0).sum())

    print("\nPacketins change statistics:")
    print(valid_changes.describe().to_string())

    print("\nExample packetins time-series records:")
    print(
        packetin_time.head(30).to_string(index=False)
    )

    # Check whether packetins is monotonic for each switch.
    print("\nPacketins monotonicity by switch:")

    for switch, group in packetin_time.groupby("switch"):

        values = group["packetins_mean"].dropna()

        increasing = values.is_monotonic_increasing
        decreasing = values.is_monotonic_decreasing

        print(
            f"Switch {switch}: "
            f"increasing={increasing}, "
            f"decreasing={decreasing}, "
            f"records={len(values)}"
        )

    # --------------------------------------------------
    # 3. PAIRFLOW FEATURE
    # --------------------------------------------------

    section("3. PAIRFLOW FEATURE INVESTIGATION")

    show_distribution(df["Pairflow"], "Pairflow")

    # Pairflow consistency for different group definitions.
    pair_groups = {
        "source + destination":
            ["src", "dst"],

        "source + destination + protocol":
            ["src", "dst", "Protocol"],

        "source + destination + protocol + switch":
            ["src", "dst", "Protocol", "switch"],

        "source + destination + protocol + timestamp":
            ["src", "dst", "Protocol", "dt"],

        "source + destination + protocol + switch + timestamp":
            ["src", "dst", "Protocol", "switch", "dt"]
    }

    print("\nPairflow group consistency:")

    for name, columns in pair_groups.items():

        result = (
            df.groupby(columns, dropna=False)["Pairflow"]
            .nunique()
        )

        total = len(result)
        consistent_count = (result == 1).sum()

        percentage = (
            consistent_count / total * 100
            if total else 0
        )

        print(
            f"{name:<60} "
            f"{consistent_count}/{total} "
            f"({percentage:.2f}%)"
        )

    # Detailed Pairflow distribution by protocol.
    section("PAIRFLOW VS PROTOCOL")

    print(
        pd.crosstab(
            df["Protocol"],
            df["Pairflow"],
            margins=True
        ).to_string()
    )

    print("\nNormalized distribution:")

    print(
        pd.crosstab(
            df["Protocol"],
            df["Pairflow"],
            normalize="index"
        ).round(4).to_string()
    )

    # Pairflow distribution by source-destination-protocol.
    section("PAIRFLOW BY SOURCE-DESTINATION-PROTOCOL")

    pair_details = (
        df.groupby(
            ["src", "dst", "Protocol"],
            dropna=False
        )
        .agg(
            records=("Pairflow", "size"),
            pairflow_values=("Pairflow", "nunique"),
            pairflow_value=("Pairflow", "first"),
            switches=("switch", "nunique"),
            timestamps=("dt", "nunique"),
            labels=("label", "nunique")
        )
        .reset_index()
    )

    print("Total tuples:", len(pair_details))

    print(
        "Tuples with consistent Pairflow:",
        (pair_details["pairflow_values"] == 1).sum()
    )

    print(
        "Tuples with inconsistent Pairflow:",
        (pair_details["pairflow_values"] > 1).sum()
    )

    print("\nPairflow tuple table:")
    print(
        pair_details.sort_values(
            ["Protocol", "src", "dst"]
        ).head(60).to_string(index=False)
    )

    # --------------------------------------------------
    # 4. RELATIONSHIP WITH LABEL
    # --------------------------------------------------

    section("4. FLOWS, PACKETINS AND PAIRFLOW BY LABEL")

    for feature in ["flows", "packetins", "Pairflow"]:

        print(f"\n{feature} by label:")
        print(
            df.groupby("label")[feature]
            .agg([
                "count",
                "mean",
                "median",
                "min",
                "max"
            ])
            .round(3)
            .to_string()
        )

    print("\nPairflow vs label:")
    print(
        pd.crosstab(
            df["Pairflow"],
            df["label"],
            normalize="index"
        ).round(4).to_string()
    )

    # --------------------------------------------------
    # 5. EXPORT USEFUL DIAGNOSTIC TABLES
    # --------------------------------------------------

    output_dir = Path("results/investigations")
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_comparison.to_csv(
        output_dir / "d1_flows_group_analysis.csv",
        index=False
    )

    packetin_time.to_csv(
        output_dir / "d1_packetins_time_analysis.csv",
        index=False
    )

    pair_details.to_csv(
        output_dir / "d1_pairflow_tuple_analysis.csv",
        index=False
    )

    section("INVESTIGATION COMPLETED")

    print("Diagnostic CSV files saved in:")
    print(output_dir.resolve())

    print("\nFiles created:")
    print("1. d1_flows_group_analysis.csv")
    print("2. d1_packetins_time_analysis.csv")
    print("3. d1_pairflow_tuple_analysis.csv")


if __name__ == "__main__":
    main()
