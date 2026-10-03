from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "verified_627A.csv")
assert len(df) == 8
assert (df[["Open", "High", "Low", "Close"]] > 0).all().all()
assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
assert int(df.loc[df["Date"].eq("2026-10-02"), "Close"].iloc[0]) == 1778
assert int(df.loc[df["Date"].eq("2026-09-29"), "High"].iloc[0]) == 2507
print("verified_627A.csv checks passed")
