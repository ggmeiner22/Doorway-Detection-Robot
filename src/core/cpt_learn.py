from __future__ import annotations
import csv
from pathlib import Path
from collections import Counter
from .config import LOCATIONS, FEATURES, POMDP_FEATURES

def get_possible_values_pomdp():
    values = {}
    for f in POMDP_FEATURES:
        if f.startswith("IR"):
            values[f] = list(range(12))
        elif f.startswith("PIDP") or f.startswith("PIDI") or f.startswith("PIDD"):
            values[f] = list(range(10))
        elif f.startswith("BI"):
            values[f] = [0, 1]  # For False/True
        elif f.startswith("ODO"):
            values[f] = list(range(6))
        elif f == "Reward":
            values[f] = [0, 1]
    return values

def get_generic_label(specific_label: str) -> str:
    if "Wall_End" in specific_label: return "Wall_End"
    if specific_label.startswith("Wall"): return "Wall"
    if specific_label.startswith("Door_Start"): return "Door_Start"
    if specific_label.startswith("Door_Passed"): return "Door_Passed"
    if specific_label.startswith("Door"): return "Door"
    return "Wall"

def learn_cpts_from_rows_pomdp(rows: list[dict], out_dir: Path, smoothing: float = 1.0):
    if not rows: raise RuntimeError("No rows to learn from.")

    values = get_possible_values_pomdp()

    # 1. Learn Prior over FULL state space (Wall_0, Door_1, etc.)
    # This ensures the POMDP knows all valid states.
    prior = Counter(r["location"] for r in rows)
    total = sum(prior.values()) + smoothing*len(LOCATIONS)
    prior_sm = {loc: (prior.get(loc,0)+smoothing)/total for loc in LOCATIONS}

    # 2. Learn Conditional Probs for GENERIC states (Wall, Door, etc.)
    # We aggregate counts from 'Wall_0', 'Wall_1' -> 'Wall'
    GENERIC_LOCATIONS = ["Wall", "Door_Start", "Door", "Door_Passed", "Wall_End"]
    
    cond = {f: {gloc: Counter() for gloc in GENERIC_LOCATIONS} for f in POMDP_FEATURES}
    count_gloc = Counter()

    for r in rows:
        loc = r["location"]
        gloc = get_generic_label(loc) # Map specific -> generic
        count_gloc[gloc] += 1
        
        for f in POMDP_FEATURES:
            val_str = r[f]
            if f == "Reward":
                val = int(val_str)
            elif f.startswith("BI"):
                val = 1 if val_str == 'True' else 0
            else:
                val = float(val_str)
            cond[f][gloc][val] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write Prior (Full States)
    with (out_dir/"prior_location.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g); w.writerow(["location","p"])
        for loc in LOCATIONS: w.writerow([loc, prior_sm[loc]])

    # Write CPTs (Generic States)
    for f in POMDP_FEATURES:
        with (out_dir/f"cpt_{f}.csv").open("w", newline="", encoding="utf-8") as g:
            w = csv.writer(g); w.writerow(["value"] + GENERIC_LOCATIONS)
            for v in values[f]:
                # Calculate P(Observation | GenericState)
                probs = []
                for gloc in GENERIC_LOCATIONS:
                    denom = count_gloc.get(gloc, 0)
                    if smoothing == 0:
                        if denom == 0:
                            p = 1.0 / len(values[f]) if len(values[f]) > 0 else 0
                        else:
                            p = cond[f][gloc].get(v, 0) / denom
                    else:
                        denom += smoothing*len(values[f])
                        num = cond[f][gloc].get(v, 0) + smoothing
                        p = num / denom
                    probs.append(p)
                
                w.writerow([v] + probs)
                
    print(f"[train] CPTs written → {out_dir.resolve()}")


def get_possible_values():
    values = {}
    for f in FEATURES:
        if f.startswith("IR"):
            values[f] = list(range(12))
        elif f.startswith("PIDP") or f.startswith("PIDI") or f.startswith("PIDD"):
            values[f] = list(range(10))
        elif f.startswith("BI"):
            values[f] = [0, 1]  # For False/True
    return values

def learn_cpts_from_rows(rows: list[dict], out_dir: Path, smoothing: float = 1.0):
    if not rows: raise RuntimeError("No rows to learn from.")

    values = get_possible_values()

    prior = Counter(r["location"] for r in rows)
    total = sum(prior.values()) + smoothing*len(LOCATIONS)
    prior_sm = {loc: (prior.get(loc,0)+smoothing)/total for loc in LOCATIONS}

    cond = {f: {loc: Counter() for loc in LOCATIONS} for f in FEATURES}
    count_loc = Counter()
    for r in rows:
        loc = r["location"]; count_loc[loc]+=1
        for f in FEATURES:
            val_str = r[f]
            if f.startswith("BI"):
                val = 1 if val_str == 'True' else 0
            else:
                val = float(val_str)
            cond[f][loc][val] += 1

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


def learn_cpts_from_csv(csv_path: Path, out_dir: Path, smoothing: float = 1.0):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows: raise RuntimeError("No rows in collected CSV.")
    learn_cpts_from_rows(rows, out_dir, smoothing)
