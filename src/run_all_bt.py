#!/usr/bin/env python3
"""Orchestrator: collect -> train CPTs -> control (Create3 over BLE).

Run with:
    python -m src.run_all_bt
"""
import time, threading
from irobot_edu_sdk.robots import event
from .config import DATA_CSV, CPTS_DIR, FEATURES, DT, ROBOT_NAME, ensure_dirs, DOOR_STATES
from .robot_io import get_robot
from .cpt_learn import learn_cpts_from_csv
from .control import warmup_autolog, run_controller

robot = get_robot(ROBOT_NAME)
_connected = False
print("[run] Using event-driven SDK loop")

@event(robot.when_play)
async def _run(_r):
    global _connected
    print(f"[run] Connected to {ROBOT_NAME}")
    _connected = True
    ensure_dirs()

    # Phase 1: WALL-FOLLOW + AUTO-LOG (no keys)
    await warmup_autolog(robot, data_csv=DATA_CSV)  # uses WARMUP_SECONDS from config

    # Optional Phase 1: collect & label (farms lots of data w/o doing the BN control afterwards)
    # await collect_auto(robot, data_csv=DATA_CSV, dt=DT)

    # Phase 2: learn CPTs
    learn_cpts_from_csv(DATA_CSV, CPTS_DIR, smoothing=1.0)

    # Phase 3: control with BN belief
    await run_controller(robot, cpts_dir=CPTS_DIR, door_states=DOOR_STATES, dt=DT)

if __name__ == "__main__":
    def heartbeat():
        while not _connected:
            print("[run] searching… (ensure robot is on, paired, and name/MAC matches)")
            time.sleep(2)
    threading.Thread(target=heartbeat, daemon=True).start()
    robot.play()
