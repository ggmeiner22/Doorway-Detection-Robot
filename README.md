# Doorway_Detection-Robot

Wall-following robot with **sensor fusion** (Naive Bayes) and **PID** control.  
You learn **CPTs** from measurements, then fuse live/batched readings to estimate:

- probability you **just passed a door** (and a temporal “~10 cm ago” helper)
- **distance from wall** as a probability distribution → expected distance for PID

The repo also includes a small demo that produces plots and a CSV you can open in Excel.

---

## Repo layout

```text
Doorway_Detection-Robot/
├── cpts/
│   ├── cpt_IR1.csv          # backup / symmetry feature
│   ├── cpt_IR5.csv          # distance estimation (right sensor)
│   └── prior_location.csv
├── data/
│   └── measurements_v2.csv
├── docs/
│   ├── collectData.py        # Old groups work EVENTUALLY REMOVE
│   ├── images/
│   │   ├── network.png
│   │   ├── posterior_rep.png
│   │   └── timeseries.png
│   └── index.html            # Project Website
├── LICENSE
├── README.md
├── requirements.txt
└── src/
    ├── belief_network.py
    ├── collect.py
    ├── config.py
    ├── control.py
    ├── cpt_learn.py
    ├── data
    │   └── measurements_live.csv
    ├── __init__.py
    ├── pid.py
    ├── robot_io.py
    ├── run_all_bt.py
    └── tests/
          └── test_drive.py
```

---

## 0) Prereqs

- Python 3.10+ (3.11 OK)
- (Recommended) VS Code with **Python**, **Pylance**, and **Remote – WSL** extensions if you use WSL.
  - Note: WSL does not support BLE

---

## 1) Setup & Execution (venv + deps)

```bash
cd Doorway-Detection-Robot
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# To execute
python -m src.run_all_bt
```

---

## 2) What the robot does, step by step

### A) Connect
- The console prints “searching…” until the Create 3 connects over Bluetooth.
- As soon as it’s connected you’ll see “Connected to …” and Phase 1 begins.

### B) Warm-up wall-following + auto-logging
- **Immediately starts** PID wall-following at **≈ 60 cm** from the right wall.  
- **Drives forward** at a slow speed; gently steers to hold the setpoint.  
- Every ~0.1 s it reads the right IR sensor, converts to distance, and **auto-labels** using the trend:

  | Condition | LED Color | Meaning |
  |------------|------------|----------|
  | **Wall** | 🟢 Green | Normal wall-follow distance |
  | **Door_Start** | 🟡 Yellow | Sustained rise → possible doorway start |
  | **Door** | 🔵 Blue | Open region detected |
  | **Door_Passed** | 🟣 Magenta | Doorway just passed (one sample) |

- It writes labeled rows to `data/measurements_live.csv`.
- Runs for **`WARMUP_SECONDS`** (default = 9 s) unless changed in the config file.

### C) Learn CPTs (Bayesian Network)
- Trains from the just-collected measurements.  
- **Writes:**
  - `cpts/prior_location.csv`
  - `cpts/cpt_IR1.csv`, `cpts/cpt_IR5.csv`
- These are **Excel-friendly CSVs** for easy visualization or analysis.

### D) BN-assisted wall-following (live inference)
- Continues **PID wall-following** after training.
- Each control cycle:

  - Computes **belief** over `Location`  
    *(Wall / Door_Start / Door / Door_Passed)*.
  - Computes **P(door just passed 10 cm ago)** by looking ~10 samples back.
  - Computes **distance-from-wall distribution** over 10 cm bins  
    (`p_distance_bins`).

- **Behavior rule:**
  - If `door_now` or `door_10cm_ago` > 0.6 → LED 🟣 **magenta**, **slow down briefly**.
  - Otherwise → LED 🟢 **green**, continue at normal speed.

- Console prints distance, bins, door probabilities, and wheel speeds continuously.

---

## 3) Files you will see
`data/measurements_live.csv` — auto-labeled training data from warm-up.

`cpts/*.csv` — learned BN tables used by control.

---

## 4) Stop / rerun
**Ctrl+C** to stop.

On later runs, you can skip warm-up and reuse CPTs (if you enable that setting); otherwise it will re-collect and retrain.

---

## 5) License

This project is released under the **MIT License**.  

You are free to use, modify, and distribute this code, but attribution is appreciated.  
Replace the sample data & CPTs with your own robot measurements for your own use.
