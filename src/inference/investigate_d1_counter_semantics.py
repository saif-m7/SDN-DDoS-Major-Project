
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("data/raw/dataset1/dataset_sdn.csv")
OUTPUT_DIR = Path("results/investigations")


def section(title):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)


def consistency_report(df, group_columns, feature):
    result = (
        df.groupby(group_columns, dropna=False)[feature]
        .nunique()
    )

    total = len(result)
    consistent = (result == 1).sum()

    return {
        "groups": total,
        "consistent": int(consistent),
        "inconsistent": int(total - consistent),
        "percentage": (
            consistent / total * 100 if total else 0
        )
    }


def print_consistency(df, feature, groupings):

    print(f"\nFeature: {feature}")

    for name, columns in groupings.items():

        result = consistency_report(
            df,
            columns,
            feature
        )

        print(
            f"{name:<55} "
            f"{result['consistent']:>7}/{result['groups']:<7} "
            f"{result['percentage']:>7.2f}%"
        )


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    required = [
        "dt", "switch", "src", "dst", "Protocol",
        "port_no", "flows", "packetins", "Pairflow",
        "pktcount", "bytecount", "label"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_columns = [
        "dt", "switch", "port_no", "flows",
        "packetins", "Pairflow", "pktcount",
        "bytecount", "label"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["dt", "switch", "flows", "packetins", "Pairflow"]
    ).copy()

    df = df.sort_values(
        ["switch", "dt"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("1. FLOW COUNT VS POSSIBLE GROUP DEFINITIONS")

    groupings = {
        "timestamp + switch":
            ["dt", "switch"],

        "timestamp + switch + protocol":
            ["dt", "switch", "Protocol"],

        "timestamp + switch + port":
            ["dt", "switch", "port_no"],

        "timestamp + switch + source":
            ["dt", "switch", "src"],

        "timestamp + switch + destination":
            ["dt", "switch", "dst"],

        "timestamp + switch + source + destination":
            ["dt", "switch", "src", "dst"],

        "timestamp + switch + source + destination + protocol":
            ["dt", "switch", "src", "dst", "Protocol"],

        "timestamp + switch + source + destination + protocol + port":
            ["dt", "switch", "src", "dst", "Protocol", "port_no"]
    }

    flow_group_rows = []

    for name, columns in groupings.items():

        counts = (
            df.groupby(columns, dropna=False)
            .size()
            .rename("record_count")
            .reset_index()
        )

        # Number of distinct tuples within each timestamp/switch group.
        if name == "timestamp + switch":
            unique_tuples = (
                df.groupby(["dt", "switch"], dropna=False)
                .apply(
                    lambda g: len(
                        g[["src", "dst", "Protocol", "port_no"]]
                        .drop_duplicates()
                    ),
                    include_groups=False
                )
                .rename("unique_tuple_count")
                .reset_index()
            )

            flow_values = (
                df.groupby(["dt", "switch"], dropna=False)
                ["flows"].first()
                .reset_index()
            )

            comparison = flow_values.merge(
                unique_tuples,
                on=["dt", "switch"]
            )

            comparison["difference"] = (
                comparison["flows"]
                - comparison["unique_tuple_count"]
            )

            print("\nFlows vs unique 4-field tuples:")
            print("Exact matches:",
                  (comparison["difference"] == 0).sum())
            print("Total groups:", len(comparison))
            print(
                "Correlation:",
                comparison[
                    ["flows", "unique_tuple_count"]
                ].corr().iloc[0, 1]
            )

            comparison.to_csv(
                OUTPUT_DIR / "d1_flows_tuple_comparison.csv",
                index=False
            )

        print(f"\n{name}: {len(counts)} groups")

    # Test whether flows is related to group record count.
    base_groups = (
        df.groupby(["dt", "switch"], dropna=False)
        .agg(
            flows=("flows", "first"),
            record_count=("flows", "size"),
            unique_src=("src", "nunique"),
            unique_dst=("dst", "nunique"),
            unique_protocol=("Protocol", "nunique"),
            unique_port=("port_no", "nunique"),
            unique_src_dst=(
                "src",
                "nunique"
            )
        )
        .reset_index()
    )

    print("\nFlows relationship with group measurements:")
    print(
        base_groups[
            [
                "flows", "record_count", "unique_src",
                "unique_dst", "unique_protocol", "unique_port"
            ]
        ].corr().round(4).to_string()
    )

    print("\nFlows compared with record count:")
    print(
        base_groups[
            ["flows", "record_count"]
        ].describe().round(3).to_string()
    )

    # --------------------------------------------------
    # 2. PACKETINS COUNTER AND RESET ANALYSIS
    # --------------------------------------------------

    section("2. PACKETINS COUNTER AND RESET ANALYSIS")

    packetins = (
        df.groupby(["switch", "dt"], as_index=False)
        .agg(
            packetins=("packetins", "first"),
            unique_values=("packetins", "nunique"),
            records=("packetins", "size")
        )
        .sort_values(["switch", "dt"])
    )

    packetins["previous"] = (
        packetins.groupby("switch")["packetins"].shift(1)
    )

    packetins["change"] = (
        packetins["packetins"] - packetins["previous"]
    )

    packetins["reset_candidate"] = (
        packetins["change"] < 0
    )

    packetins["increase_candidate"] = (
        packetins["change"] > 0
    )

    print("\nPacketins change classification:")
    print(
        packetins["change"]
        .apply(
            lambda x:
                "FIRST" if pd.isna(x)
                else "RESET/DECREASE" if x < 0
                else "INCREASE" if x > 0
                else "UNCHANGED"
        )
        .value_counts()
        .to_string()
    )

    print("\nNegative changes by switch:")
    print(
        packetins.groupby("switch")["reset_candidate"]
        .sum()
        .to_string()
    )

    print("\nReset candidate examples:")
    print(
        packetins[
            packetins["reset_candidate"]
        ].head(40).to_string(index=False)
    )

    print("\nPacketins values immediately before and after decreases:")

    reset_rows = packetins.index[
        packetins["reset_candidate"]
    ]

    examples = []

    for idx in reset_rows[:20]:

        start = max(0, idx - 1)
        end = min(len(packetins), idx + 2)

        examples.append(packetins.iloc[start:end])

    if examples:
        print(
            pd.concat(examples)
            .drop_duplicates()
            .to_string(index=False)
        )

    # Check whether packetins values repeat across switches.
    packetin_switch = (
        df.groupby(["dt", "switch"], as_index=False)
        .agg(packetins=("packetins", "first"))
    )

    packetin_timestamp = (
        packetin_switch.groupby("dt")
        .agg(
            switches=("switch", "nunique"),
            unique_packetins=("packetins", "nunique"),
            min_packetins=("packetins", "min"),
            max_packetins=("packetins", "max")
        )
        .reset_index()
    )

    print("\nPacketins consistency across switches at same timestamp:")
    print(
        packetin_timestamp["unique_packetins"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------
    # 3. PAIRFLOW TEMPORAL BEHAVIOUR
    # --------------------------------------------------

    section("3. PAIRFLOW TEMPORAL BEHAVIOUR")

    pair_columns = [
        "src", "dst", "Protocol"
    ]

    pair_time = (
        df.groupby(
            pair_columns + ["dt"],
            dropna=False,
            as_index=False
        )
        .agg(
            Pairflow=("Pairflow", "first"),
            unique_pairflow=("Pairflow", "nunique"),
            switches=("switch", "nunique"),
            records=("Pairflow", "size"),
            labels=("label", "nunique")
        )
        .sort_values(pair_columns + ["dt"])
    )

    pair_time["previous_pairflow"] = (
        pair_time.groupby(pair_columns)["Pairflow"].shift(1)
    )

    pair_time["pairflow_changed"] = (
        pair_time["Pairflow"]
        != pair_time["previous_pairflow"]
    )

    pair_time.loc[
        pair_time["previous_pairflow"].isna(),
        "pairflow_changed"
    ] = False

    print("\nPairflow changes across consecutive observed timestamps:")
    print(
        pair_time.groupby("Protocol")["pairflow_changed"]
        .agg(["count", "sum", "mean"])
        .round(4)
        .to_string()
    )

    print("\nTuples with Pairflow changes:")
    changed_pairs = (
        pair_time.groupby(pair_columns)
        .agg(
            timestamps=("dt", "nunique"),
            pairflow_values=("Pairflow", "nunique"),
            switches=("switches", "max")
        )
        .reset_index()
    )

    changed_pairs = changed_pairs[
        changed_pairs["pairflow_values"] > 1
    ]

    print("Number of changing tuples:", len(changed_pairs))
    print(changed_pairs.head(30).to_string(index=False))

    print("\nPairflow changes with switch involvement:")
    print(
        pair_time[
            pair_time["pairflow_changed"]
        ].head(40).to_string(index=False)
    )

    # --------------------------------------------------
    # 4. SOURCE-DESTINATION DIRECTION ANALYSIS
    # --------------------------------------------------

    section("4. FORWARD AND REVERSE TRAFFIC PAIRS")

    forward = df[
        ["src", "dst", "Protocol", "Pairflow"]
    ].drop_duplicates()

    reverse = forward.rename(
        columns={
            "src": "dst",
            "dst": "src",
            "Pairflow": "reverse_pairflow"
        }
    )

    direction = forward.merge(
        reverse,
        on=["src", "dst", "Protocol"],
        how="left"
    )

    direction["reverse_exists"] = (
        direction["reverse_pairflow"].notna()
    )

    direction["same_pairflow"] = (
        direction["Pairflow"]
        == direction["reverse_pairflow"]
    )

    print("Unique directed tuples:", len(direction))
    print(
        "Tuples with reverse direction:",
        direction["reverse_exists"].sum()
    )

    print(
        "Reverse pairs with same Pairflow:",
        direction[
            direction["reverse_exists"]
        ]["same_pairflow"].sum()
    )

    print("\nDirection comparison examples:")
    print(direction.head(40).to_string(index=False))

    # --------------------------------------------------
    # 5. PAIRFLOW VS LABEL AND SWITCH
    # --------------------------------------------------

    section("5. PAIRFLOW VS LABEL AND SWITCH")

    print("\nPairflow by switch:")
    print(
        pd.crosstab(
            df["switch"],
            df["Pairflow"],
            normalize="index"
        ).round(4).to_string()
    )

    print("\nPairflow by label and protocol:")
    print(
        df.groupby(["Protocol", "label"])["Pairflow"]
        .agg(["count", "mean"])
        .round(4)
        .to_string()
    )

    # --------------------------------------------------
    # 6. SAVE OUTPUTS
    # --------------------------------------------------

    section("6. EXPORT RESULTS")

    base_groups.to_csv(
        OUTPUT_DIR / "d1_flows_measurement_comparison.csv",
        index=False
    )

    packetins.to_csv(
        OUTPUT_DIR / "d1_packetins_reset_analysis.csv",
        index=False
    )

    pair_time.to_csv(
        OUTPUT_DIR / "d1_pairflow_temporal_analysis.csv",
        index=False
    )

    direction.to_csv(
        OUTPUT_DIR / "d1_pairflow_direction_analysis.csv",
        index=False
    )

    print("Saved diagnostic files to:", OUTPUT_DIR.resolve())

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
