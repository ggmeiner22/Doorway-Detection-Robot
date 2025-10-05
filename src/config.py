from pathlib import Path
import os

ROOT        = Path(__file__).resolve().parent
DATA_CSV    = ROOT / "data" / "measurements_live.csv"
CPTS_DIR    = ROOT / "cpts"
FEATURES    = ["IR1", "IR5"]
LOCATIONS   = ["Wall","Door_Start","Door","Door_Passed"]
DOOR_STATES = ["Door_Passed"]

RIGHT_IR_IDX = 6
HZ           = 10.0
DT           = 1.0 / HZ
SETPOINT_CM  = 60.0
FORWARD      = 10
MAX_W, MIN_W = 21, 1

# Prefer explicit MAC via env; fallback to default advertised name
#ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-88FA7A7E3FCC461E8B675C")
ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-030F9BF3B40449DC94031C")

def ensure_dirs():
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    CPTS_DIR.mkdir(parents=True, exist_ok=True)
