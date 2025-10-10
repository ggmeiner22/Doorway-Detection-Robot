from __future__ import annotations
import csv
from pathlib import Path
from collections import Counter
from .config import LOCATIONS, FEATURES

def get_possible_values():
    values = {}
    for f in FEATURES:
        if f.startswith("IR"):
            values[f] = list(range(10))
        elif f.startswith("PIDP"):
            values[f] = [0.0, 0.5, 1.0, 1.5, 2.0]
        elif f.startswith("PIDI"):
            values[f] = [-1, 0, 1]
        elif f.startswith("PIDD"):
            values[f] = [-1, 0, 1]
        elif f.startswith("BI"):
            values[f] = [0, 1]  # For False/True
    return values

def learn_cpts_from_csv(csv_path: Path, out_dir: Path, smoothing: float = 1.0):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows: raise RuntimeError("No rows in collected CSV.")

    values = get_possible_values()

    prior = Counter(r["location"] for r in rows)
    total = sum(prior.values()) + smoothing*len(LOCATIONS)
    prior_sm = {loc: (prior.get(loc,0)+smoothing)/total for loc in LOCATIONS}

    cond = {f: {loc: Counter() for loc in LOCATIONS} for f in FEATURES}
    count_loc = Counter()
    for r in rows:
        loc = r["location"]; count_loc[loc]+=1
        for f in FEATURES: cond[f][loc][float(r[f])]+=1

    for f in FEATURES:
        for loc in LOCATIONS:
            denom = count_loc.get(loc,0)
            if smoothing == 0:
                if denom == 0:
                    num_values = len(values[f])
                    for v in values[f]:
                        cond[f][loc][v] = 1.0 / num_values if num_values > 0 else 0
                    continue
                for v in values[f]:
                    cond[f][loc][v] = cond[f][loc].get(v, 0) / denom
            else:
                denom += smoothing*len(values[f])
                for v in values[f]:
                    num = cond[f][loc].get(v,0) + smoothing
                    cond[f][loc][v] = num/denom

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/"prior_location.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g); w.writerow(["location","p"])
        for loc in LOCATIONS: w.writerow([loc, prior_sm[loc]])
    for f in FEATURES:
        with (out_dir/f"cpt_{f}.csv").open("w", newline="", encoding="utf-8") as g:
            w = csv.writer(g); w.writerow(["value"] + LOCATIONS)
            for v in values[f]:
                w.writerow([v] + [cond[f][loc].get(v, 0.0) for loc in LOCATIONS])
    print(f"[train] CPTs written → {out_dir.resolve()}")