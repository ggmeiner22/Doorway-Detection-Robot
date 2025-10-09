from core.cpt_learn import learn_cpts_from_csv
from core.config import DATA_CSV, CPTS_DIR

if __name__ == "__main__":
    learn_cpts_from_csv(DATA_CSV, CPTS_DIR, smoothing=1.0)