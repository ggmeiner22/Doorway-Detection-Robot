from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "measurements_v2.csv"
CPTS_DIR = ROOT / "cpts"
IMG_DIR  = ROOT / "docs" / "images"   # default output directory
IMG_DIR.mkdir(parents=True, exist_ok=True)

from src.belief_network import belief, door_passed_10cm_ago

def expected_distance(b, mapping):
    return sum(p * mapping.get(l, 7.0) for l, p in zip(b["locations"], b["p_location"]))

def main():
    # ---- CLI args
    parser = argparse.ArgumentParser(description="Doorway detection demo")
    parser.add_argument("--data", type=str, default=str(DATA_CSV),
                        help="Path to measurements CSV (default: data/measurements_v2.csv)")
    parser.add_argument("--cpts", type=str, default=str(CPTS_DIR),
                        help="Directory containing learned CPTs (default: cpts/)")
    parser.add_argument("--outdir", type=str, default=str(IMG_DIR),
                        help="Directory for outputs (CSV + PNGs). Default: docs/images/")
    args = parser.parse_args()

    # ---- ensure output dir exists
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load dataset
    data = pd.read_csv(args.data)

    # Example slice (pick region of interest)
    start, end = 350, 600
    sub = data.iloc[start:end].reset_index(drop=True)

    # Mapping: Location -> representative distance (cm)
    loc_to_cm = {
        "wall_close": 5.0, "wall_mid": 6.5, "wall_far": 9.0,
        "door_start": 7.5, "door_passed": 9.5,
    }

    beliefs, exps, pnow, pago = [], [], [], []
    for _, row in sub.iterrows():
        reading = {"IR1": int(row["IR3"]), "IR5": int(row["IR6"])}
        b = belief(reading, {"cpts_dir": args.cpts, "door_states": ["door_passed"]})
        beliefs.append(b)
        exps.append(expected_distance(b, loc_to_cm))
        pnow.append(b["p_door_passed"])
        pago.append(door_passed_10cm_ago(beliefs, cm_per_step=1.0))

    out = pd.DataFrame({
        "t": range(len(sub)),
        "expected_cm": exps,
        "p_door_now": pnow,
        "p_door_10cm_ago": pago
    })
    # Save CSV
    out.to_csv(outdir / "belief_timeseries.csv", index=False)
    
    # Posterior (representative)
    if not beliefs:
        print("No beliefs computed; check your data slice or feature names."); return
    mid = len(beliefs) // 2
    rep = beliefs[mid]

    plt.figure()
    plt.bar(rep["locations"], rep["p_location"])
    plt.title("Posterior over Location (representative)")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(outdir / "posterior_rep.png")
    
    # Time series
    plt.figure()
    plt.plot(out["t"], out["expected_cm"], label="Expected distance (cm)")
    plt.plot(out["t"], out["p_door_now"], label="p(door now)")
    plt.plot(out["t"], out["p_door_10cm_ago"], label="p(door 10cm ago)")
    plt.xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "timeseries.png")
    
    print(f"Wrote {outdir/'belief_timeseries.csv'} and {outdir/'posterior_rep.png'}, {outdir/'timeseries.png'}")


if __name__ == "__main__":
    main()
