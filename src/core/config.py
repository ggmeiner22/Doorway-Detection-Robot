from pathlib import Path
import os

ROOT         = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_CSV     = PROJECT_ROOT / ".." / "data" / "synthetic_measurements.csv"
CPTS_DIR     = PROJECT_ROOT / ".." / "cpts"
FEATURES     = [f"IR{i}" for i in range(1, 10)] + \
             [f"PIDP{i}" for i in range(1, 10)] + \
             [f"PIDD{i}" for i in range(1, 10)] + \
             [f"PIDI{i}" for i in range(1, 10)] + \
             [f"BI{i}" for i in range(1, 10)]
LOCATIONS    = ["Wall","Door_Start","Door","Door_Passed"]
DOOR_STATES  = ["Door_Passed"]

RIGHT_IR_IDX  = 6
HZ            = 10.0
DT            = 1.0 / HZ
SETPOINT_CM   = 60.0
FORWARD       = 10
MAX_W, MIN_W  = 21, 1

# --- warm-up control + auto logging ---
WARMUP_SECONDS = 9.0        # time to collect measurements while wall-following
AUTO_K_RISE = 3.0           # sigma thresholds for door detection during warmup
AUTO_K_FALL = 2.0
AUTO_MIN_RISE_SAMPLES = 4
AUTO_MIN_DOOR_SAMPLES = 5
AUTO_REFRACTORY_STEPS = 10
AUTO_EWMA_ALPHA = 0.12
AUTO_EWVAR_ALPHA = 0.08

# control & detection
DOOR_TRIGGER = 0.60          # slows + magenta if belief exceeds this
CM_PER_STEP = 1.0            # for "10 cm ago" (≈10 cycles at 10 Hz)


# Prefer explicit MAC via env; fallback to default advertised name
ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-030")
#ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot_25")
#ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-88FA7A7E3FCC461E8B675C")
#ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-030F9BF3B40449DC94031C")

def ensure_dirs():
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    CPTS_DIR.mkdir(parents=True, exist_ok=True)
