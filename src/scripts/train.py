import csv
from src.core.cpt_learn import learn_cpts_from_rows
from src.core.config import DATA_CSV, CPTS_DIR

if __name__ == "__main__":
    data_dir = DATA_CSV.parent
    # Include all measurement CSVs
    all_csvs = list(data_dir.glob("measurements_*.csv"))

    all_rows = []
    print("[train] Loading data files...")
    for csv_path in all_csvs:
        if not csv_path.exists():
            continue
        print(f"  - {csv_path.name}")
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            all_rows.extend(list(rdr))

    if not all_rows:
        print("[train] No data found to train on. Exiting.")
        exit()

    print(f"[train] Learning CPTs from {len(all_rows)} total rows across {len(all_csvs)} file(s)...")
    learn_cpts_from_rows(all_rows, CPTS_DIR, smoothing=0.0)
