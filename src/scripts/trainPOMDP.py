import csv
import shutil
from src.core.cpt_learn import learn_cpts_from_rows_pomdp
from src.core.config import DATA_CSV, CPTS_DIR

if __name__ == "__main__":
    data_dir = DATA_CSV.parent
    # Include all measurement CSVs
    all_csvs = list(data_dir.glob("measurements_pomdp_*.csv"))

    all_rows = []
    print("[trainPOMDP] Loading data files...")
    for csv_path in all_csvs:
        if not csv_path.exists():
            continue
        print(f"  - {csv_path.name}")
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            all_rows.extend(list(rdr))

    if not all_rows:
        print("[trainPOMDP] No data found to train on. Exiting.")
        exit()

    print(f"[trainPOMDP] Learning CPTs from {len(all_rows)} total rows across {len(all_csvs)} file(s)...")
    learn_cpts_from_rows_pomdp(all_rows, CPTS_DIR, smoothing=0.0)

    # --- Create zip archives for website download ---
    print("[trainPOMDP] Creating zip archives for website...")
    
    # Define paths
    project_root = CPTS_DIR.parent
    data_exports_dir = project_root / 'docs' / 'data_exports'

    # Create the exports directory
    data_exports_dir.mkdir(exist_ok=True)

    # Archive the CPTs directory
    cpts_archive_path = data_exports_dir / 'cpts_pomdp'
    shutil.make_archive(str(cpts_archive_path), 'zip', str(project_root), 'cpts')
    print(f"[trainPOMDP] CPTs archive created at {str(cpts_archive_path)}.zip")

    # Archive the data directory
    data_archive_path = data_exports_dir / 'data_pomdp'
    shutil.make_archive(str(data_archive_path), 'zip', str(project_root), 'data')
    print(f"[trainPOMDP] Data archive created at {str(data_archive_path)}.zip")
