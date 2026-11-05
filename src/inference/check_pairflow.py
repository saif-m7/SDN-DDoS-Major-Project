
import pandas as pd

# Load Dataset 1
df = pd.read_csv("data/raw/dataset1/dataset_sdn.csv")

# Keep unique flow records
keys = ["dt", "switch", "Protocol", "src", "dst"]

unique_records = df[keys + ["Pairflow"]].drop_duplicates().copy()

# Build reverse-flow keys
reverse_records = unique_records[keys].copy()
reverse_records = reverse_records.rename(
    columns={"src": "dst", "dst": "src"}
)

reverse_records["reverse_exists"] = True

# Check whether reverse flow exists
result = unique_records.merge(
    reverse_records,
    on=keys,
    how="left"
)

result["reverse_exists"] = (
    result["reverse_exists"].fillna(False).astype(bool)
)

# Overall comparison
print("\nPAIRFLOW VS REVERSE PAIR")
print(
    pd.crosstab(
        result["reverse_exists"],
        result["Pairflow"],
        margins=True
    )
)

# Protocol-wise comparison
print("\nRESULT BY PROTOCOL")
print(
    result.groupby(
        ["Protocol", "reverse_exists", "Pairflow"]
    ).size().to_string()
)

print("\nTOTAL UNIQUE RECORDS:", len(result))
