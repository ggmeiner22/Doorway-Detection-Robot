from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List

def _load_prior(cpts_dir: Path) -> dict[str, float]:
    prior = {}
    with (cpts_dir / "prior_location.csv").open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            prior[r["location"]] = float(r["p"])
    return prior

def _load_cpt(cpts_dir: Path, feature: str):
    path = cpts_dir / f"cpt_{feature}.csv"
    values = []
    table = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        locations = header[1:]
        for row in rdr:
            v = int(row[0])
            values.append(v)
            for i, loc in enumerate(locations):
                table.setdefault(loc, {})[v] = float(row[1+i])
    return locations, values, table

def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        n = len(d); return {k: 1.0/n for k in d}
    return {k: v/s for k, v in d.items()}

def belief(readings: Dict[str, int], inner_configuration: Dict) -> Dict[str, float]:
    cpts_dir = Path(inner_configuration.get("cpts_dir", "cpts"))
    door_states = set(inner_configuration.get("door_states", ["Door_Passed"]))

    prior = _load_prior(cpts_dir)
    post = dict(prior)

    for feat, v in readings.items():
        cpt_file = cpts_dir / f"cpt_{feat}.csv"
        if not cpt_file.exists():
            continue
        locs, values, table = _load_cpt(cpts_dir, feat)
        for loc in post:
            post[loc] *= table.get(loc, {}).get(v, 1e-9)

    post = _normalize(post)
    p_door = sum(post.get(s, 0.0) for s in door_states)
    
    # Loads the emission CPT file for one feature from cpts/
    def _table(feat):
        p = cpts_dir / f"cpt_{feat}.csv"
        if not p.exists(): return [], {}
        locs, values, tab = _load_cpt(cpts_dir, feat)
        return values, tab
    
    bins1, tab1 = _table("IR1")
    bins5, tab5 = _table("IR5")
    support = bins1 or bins5
    p_bin = {b: 0.0 for b in support}
    for b in support:
        like1 = sum(post.get(loc,0.0)*tab1.get(loc,{}).get(b,0.0) for loc in post) if bins1 else 0.0
        like5 = sum(post.get(loc,0.0)*tab5.get(loc,{}).get(b,0.0) for loc in post) if bins5 else 0.0
        p_bin[b] = (like1 + like5) / (2 if (bins1 and bins5) else 1)
    Z = sum(p_bin.values()) or 1.0
    p_distance_bins = {b: v/Z for b,v in p_bin.items()}
    
    return {
        "posterior": post,
        "p_door_passed": p_door,
        "p_distance_bins": p_distance_bins
    }


def door_passed_10cm_ago(history: List[Dict[str, float]], cm_per_step: float = 1.0, door_states=("Door_Passed",)):
    steps = int(round(10.0 / max(cm_per_step, 1e-6)))
    if len(history) <= steps:
        return 0.0
    idx = -steps
    entry = history[idx]
    if "posterior" in entry:
        return sum(entry["posterior"].get(s, 0.0) for s in door_states)
    return entry.get("p_door_passed", 0.0)
