from __future__ import annotations
import csv, asyncio
from irobot_edu_sdk.robots import Create3
from core.robot_io import prox_to_cm, cm_to_bin10
from core.config import RIGHT_IR_IDX, DT, FEATURES

current_label = "Wall"
quit_collect  = False

async def collect_auto(robot: Create3, *, data_csv, dt: float = DT,
                       k_rise: float = 2.5, k_fall: float = 1.5,
                       min_rise_samples: int = 3, min_door_samples: int = 4,
                       ewma_alpha: float = 0.15, ewvar_alpha: float = 0.10):
    """
    Auto-label Phase 1 using an FSM on right IR distance:
      Wall -> (sustained rise) -> Door_Start -> Door -> (fall) -> Door_Passed -> Wall
    """
    new_file = not data_csv.exists()
    f = data_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["location"] + FEATURES)

    print("[collect:auto] starting; robot will drive forward slowly")
    await robot.set_wheel_speeds(8, 8)

    # EWMA baseline + variance
    base = None
    var  = 0.0
    state = "Wall"
    rise_count = 0
    door_count = 0

    try:
        while True:
            pm = await robot.get_ir_proximity()
            if pm is None or pm.sensors is None:
                await asyncio.sleep(dt); continue
            d_cm = prox_to_cm(pm.sensors[RIGHT_IR_IDX])
            b = cm_to_bin10(d_cm)

            # Update EWMA baseline/var
            if base is None:
                base = d_cm
                var = 25.0  # ~5cm sigma^2 initial guess
            else:
                base = (1 - ewma_alpha) * base + ewma_alpha * d_cm
                # EW variance (Welford-like): var tracks squared deviation
                dev = d_cm - base
                var  = (1 - ewvar_alpha) * var + ewvar_alpha * (dev * dev)

            sigma = max(3.0, var ** 0.5)  # floor sigma for stability
            rise_thresh = base + k_rise * sigma
            fall_thresh = base + k_fall * sigma

            # FSM
            label = "Wall"
            if state == "Wall":
                if d_cm > rise_thresh:
                    rise_count += 1
                    if rise_count >= min_rise_samples:
                        state = "Door_Start"
                        label = "Door_Start"
                        door_count = 0
                        rise_count = 0
                else:
                    rise_count = 0
                    label = "Wall"

            elif state == "Door_Start":
                # first row as Door_Start, then transition into Door
                state = "Door"
                label = "Door"
                door_count = 1

            elif state == "Door":
                door_count += 1
                if d_cm <= fall_thresh and door_count >= min_door_samples:
                    state = "Door_Passed"
                    label = "Door_Passed"
                else:
                    label = "Door"

            elif state == "Door_Passed":
                # one row of Door_Passed, then go back to Wall
                state = "Wall"
                label = "Wall"

            # Log (duplicate bin for IR1/IR5 as before)
            w.writerow([label, b, b]); f.flush()
            print(f"[collect:auto] base≈{base:5.1f} σ≈{sigma:4.1f} d={d_cm:5.1f}cm bin={b}  label={label:12s}  state={state}")
            await asyncio.sleep(dt)
    finally:
        await robot.set_wheel_speeds(0, 0)
        f.close()
        print(f"[collect:auto] saved → {data_csv.resolve()}")
