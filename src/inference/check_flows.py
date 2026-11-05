
import pandas as pd

df = pd.read_csv("data/raw/dataset1/dataset_sdn.csv")

print("FLOWS SUMMARY")
print(df["flows"].describe())

print("\nUNIQUE VALUES:", df["flows"].nunique())

print("\nFLOWS BY TIMESTAMP AND SWITCH")

grouped = (
    df.groupby(["dt", "switch"])["flows"]
    .agg(["count", "min", "max", "nunique"])
)

print(grouped.head(20).to_string())

print("\nCONSISTENCY WITHIN TIMESTAMP + SWITCH")
print("Total groups:", len(grouped))
print("One unique value:", (grouped["nunique"] == 1).sum())
print("Multiple values:", (grouped["nunique"] > 1).sum())

print("\nFLOWS BY SWITCH")
print(
    df.groupby("switch")["flows"]
    .agg(["count", "min", "median", "max", "nunique"])
)
