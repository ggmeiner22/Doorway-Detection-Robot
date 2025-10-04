
import pandas as pd, json
from pathlib import Path
def learn_cpts(csv_path, smoothing=1.0, out_dir="cpts"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    loc_values = list(df["location"].unique())
    prior = (df["location"].value_counts().reindex(loc_values).fillna(0) + smoothing)
    prior = prior / prior.sum()
    prior.rename("P(Location)").reset_index().rename(columns={"index":"location"}).to_csv(out/"prior_location.csv", index=False)
    features = [c for c in df.columns if c!="location"]
    for feat in features:
        vals = sorted(df[feat].astype(int).unique())
        table = pd.crosstab(df["location"], df[feat]).reindex(index=loc_values, columns=vals, fill_value=0)
        table = (table + smoothing).div((table + smoothing).sum(axis=1), axis=0)
        table.index.name="location"
        table.to_csv(out/f"cpt_{feat}.csv")
    (out/"meta.json").write_text(json.dumps({"locations":loc_values, "features":features}))
    return str(out)
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="cpts")
    ap.add_argument("--smoothing", type=float, default=1.0)
    a = ap.parse_args(); print(learn_cpts(a.data, a.smoothing, a.out))
