#!/usr/bin/env python3
"""
ONE-SHOT PIPELINE FOR CREATE3 + BLUETOOTH

Run:
    python run_all_bt.py

Phases:
  1) Live data collection + labeling from the robot (Bluetooth)
     - keys: w=Wall, d=Door, s=Door_Start, p=Door_Passed, q=finish
     - LEDs mirror the current label
  2) Train CPTs from the collected CSV (writes ./cpts/)
  3) Control: PID wall following + Bayesian belief (uses your belief_network.py)
"""
import os, sys, csv, time, asyncio, threading
from pathlib import Path

# iRobot EDU SDK (Bluetooth)
try:
    from irobot_edu_sdk.backend.bluetooth import Bluetooth
    from irobot_edu_sdk.robots import Create3, event
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: irobot_edu_sdk.\n"
        "Activate your venv and install it:\n"
        "  Linux/macOS:  pip install irobot_edu_sdk numpy\n"
        "  Windows:      pip install irobot_edu_sdk numpy bleak-winrt\n"
    ) from e


# Your modules
from .belief_network import belief, door_passed_10cm_ago
from .pid import PID

from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "measurements_live.csv"
CPTS_DIR = ROOT / "cpts"

# ------------------ CONFIG ------------------
ROOT = Path(__file__).resolve().parent
#ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-88FA7A7E3FCC461E8B675C")
ROBOT_NAME   = os.environ.get("IROBOT_NAME", "iRobot-030F9BF3B40449DC94031C")
DATA_CSV     = ROOT / "data" / "measurements_live.csv"
CPTS_DIR     = ROOT / "cpts"
DOOR_STATES  = ["Door_Passed"]
RIGHT_IR_IDX = 6
HZ           = 10.0
DT           = 1.0 / HZ
SETPOINT_CM  = 60.0
FORWARD      = 10
MAX_W, MIN_W = 21, 1
FEATURES     = ["IR1", "IR5"]
LOCATIONS    = ["Wall","Door_Start","Door","Door_Passed"]

def ensure_dirs():
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    CPTS_DIR.mkdir(parents=True, exist_ok=True)

def prox_to_cm(raw):
    try:
        r = float(raw)
    except:
        return 999.0
    return max(5.0, min(150.0, 2000.0 / max(1.0, r)))

def cm_to_bin10(cm):
    return int(max(0, min(9, cm // 10)))

async def set_led_for_label(robot, label):
    if label == "Wall":          await robot.set_lights_on_rgb(0,255,0)
    elif label == "Door":        await robot.set_lights_on_rgb(0,0,255)
    elif label == "Door_Start":  await robot.set_lights_on_rgb(255,255,0)
    elif label == "Door_Passed": await robot.set_lights_on_rgb(255,0,255)

# Assign robot
robot = Create3(Bluetooth(ROBOT_NAME))
_connected_flag = False

# ---------- Phase 1: Collect ----------
current_label = "Wall"
quit_collect  = False

def key_reader():
    global current_label, quit_collect
    print("\\n[collect] labeling keys:  w=Wall, d=Door, s=Door_Start, p=Door_Passed, q=finish")
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            continue
        ch = ch.lower()
        if ch == 'q':
            quit_collect = True; break
        elif ch == 'w':
            current_label = "Wall"
        elif ch == 'd':
            current_label = "Door"
        elif ch == 's':
            current_label = "Door_Start"
        elif ch == 'p':
            current_label = "Door_Passed"
        print(f"[collect] label = {current_label}")

async def phase_collect(robot: Create3):
    ensure_dirs()
    new_file = not DATA_CSV.exists()
    f = DATA_CSV.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["location"] + FEATURES)

    t = threading.Thread(target=key_reader, daemon=True)
    if os.name == "posix":
        import tty, termios
        global _tty_old
        _tty_old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    t.start()

    print("[collect] driving forward slowly; press keys to change label; 'q' to finish")
    await robot.set_wheel_speeds(8, 8)
    prev_label = None
    try:
        while not quit_collect:
            if current_label != prev_label:
                await set_led_for_label(robot, current_label)
                prev_label = current_label

            #prox = (await robot.get_ir_proximity()).sensors
            #d_cm = prox_to_cm(prox[RIGHT_IR_IDX])
            prox_msg = await robot.get_ir_proximity()
            if prox_msg is None or prox_msg.sensors is None:
                await asyncio.sleep(DT)
                continue
            d_cm = prox_to_cm(prox_msg.sensors[RIGHT_IR_IDX])
            b = cm_to_bin10(d_cm)

            w.writerow([current_label, b, b]); f.flush()
            print(f"[collect] label={current_label:12s} dist≈{d_cm:5.1f}cm bin={b}")
            await asyncio.sleep(DT)
    finally:
        await robot.set_wheel_speeds(0,0)
        f.close()
        if os.name == "posix":
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _tty_old)
        print(f"[collect] saved → {DATA_CSV.resolve()}")

# ---------- Phase 2: Train CPTs ----------
def learn_cpts_from_csv(csv_path: Path, out_dir: Path, smoothing: float = 1.0):
    import json
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr: rows.append(r)
    if not rows:
        raise RuntimeError("No rows in collected CSV.")

    values = {f:set() for f in FEATURES}
    for r in rows:
        for f in FEATURES:
            values[f].add(int(r[f]))
    values = {f: sorted(list(v)) for f,v in values.items()}

    from collections import Counter
    prior = Counter(r["location"] for r in rows)
    total = sum(prior.values()) + smoothing*len(LOCATIONS)
    prior_sm = {loc: (prior.get(loc,0)+smoothing)/total for loc in LOCATIONS}

    cond = {f: {loc: Counter() for loc in LOCATIONS} for f in FEATURES}
    count_loc = Counter()
    for r in rows:
        loc = r["location"]; count_loc[loc]+=1
        for f in FEATURES: cond[f][loc][int(r[f])]+=1

    for f in FEATURES:
        for loc in LOCATIONS:
            denom = count_loc.get(loc,0) + smoothing*len(values[f])
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
                w.writerow([v] + [cond[f][loc][v] for loc in LOCATIONS])

    meta = {"locations": LOCATIONS, "features": FEATURES, "value_domain": {f: values[f] for f in FEATURES}}
    (out_dir/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] CPTs written → {out_dir.resolve()}")

# ---------- Phase 3: Control ----------
async def phase_control(robot: Create3):
    print("[control] PID + belief using ./cpts")
    controller = PID(kp=0.4, ki=0.02, kd=0.1, setpoint=SETPOINT_CM)

    await robot.set_lights_on_rgb(0,255,0)
    await robot.set_wheel_speeds(FORWARD, FORWARD)

    history = []
    try:
        while True:
            #prox = (await robot.get_ir_proximity()).sensors
            #d1 = prox_to_cm(prox[RIGHT_IR_IDX])
            pm1 = await robot.get_ir_proximity()
            if pm1 is None or pm1.sensors is None:
                await asyncio.sleep(DT); continue
            d1 = prox_to_cm(pm1.sensors[RIGHT_IR_IDX])
            await asyncio.sleep(0.02)
            #prox = (await robot.get_ir_proximity()).sensors
            #d2 = prox_to_cm(prox[RIGHT_IR_IDX])
            pm2 = await robot.get_ir_proximity()
            if pm2 is None or pm2.sensors is None:
                await asyncio.sleep(DT); continue
            d2 = prox_to_cm(pm2.sensors[RIGHT_IR_IDX])
            dist_cm = 0.5*(d1+d2)

            u = controller.update(measurement=dist_cm, dt=DT)
            L = max(min(FORWARD + u, MAX_W), MIN_W)
            R = max(min(FORWARD - u, MAX_W), MIN_W)
            await robot.set_wheel_speeds(L, R)

            reading = {"IR1": cm_to_bin10(dist_cm), "IR5": cm_to_bin10(dist_cm)}
            b = belief(reading, {"cpts_dir": str(CPTS_DIR), "door_states": DOOR_STATES})
            history.append(b)
            p_now = b.get("p_door_passed", 0.0)
            p_ago = door_passed_10cm_ago(history, cm_per_step=1.0, door_states=tuple(DOOR_STATES))

            if p_now > 0.6 or p_ago > 0.6:
                await robot.set_lights_on_rgb(255,0,255)  # magenta
                await robot.set_wheel_speeds(max(MIN_W, FORWARD*0.5), max(MIN_W, FORWARD*0.5))
            else:
                await robot.set_lights_on_rgb(0,255,0)    # green

            print(f"[control] dist≈{dist_cm:5.1f}cm bin={reading['IR1']} door_now={p_now:.2f} door_10cm_ago={p_ago:.2f} L,R=({L:.1f},{R:.1f})")
            await asyncio.sleep(DT)
    finally:
        await robot.set_wheel_speeds(0,0)
        await robot.set_lights_on_rgb(255,0,0)


# ---------- MAIN ----------
print("[run] Using event-driven SDK loop")

@event(robot.when_play)
async def _run(_robot):
    global _connected_flag
    print(f"[run] Connected to {ROBOT_NAME}")
    _connected_flag = True

    await phase_collect(robot)                     # Phase 1: collect+label
    learn_cpts_from_csv(DATA_CSV, CPTS_DIR, 1.0)   # Phase 2: train CPTs
    await phase_control(robot)                     # Phase 3: control

# Entry point (no asyncio.run here)
if __name__ == "__main__":
    # Heartbeat while searching, so the terminal isn’t silent if not connected yet
    import threading, time
    def heartbeat():
        while not _connected_flag:
            print("[run] searching… (ensure the robot is on, paired, and name/MAC matches)")
            time.sleep(2)
    threading.Thread(target=heartbeat, daemon=True).start()

    robot.play()   # starts SDK loop and invokes @_run above
