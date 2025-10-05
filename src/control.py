from __future__ import annotations
import asyncio
from irobot_edu_sdk.robots import Create3
from .pid import PID
from .robot_io import prox_to_cm, cm_to_bin10
from .belief_network import belief, door_passed_10cm_ago
from .config import (
    DT, SETPOINT_CM, FORWARD, MAX_W, MIN_W, RIGHT_IR_IDX,
    WARMUP_SECONDS, AUTO_K_RISE, AUTO_K_FALL, AUTO_MIN_RISE_SAMPLES, AUTO_MIN_DOOR_SAMPLES,
    AUTO_REFRACTORY_STEPS, AUTO_EWMA_ALPHA, AUTO_EWVAR_ALPHA
)


def _prox_to_cm(raw):
    try: r = float(raw)
    except Exception: return 999.0
    return max(5.0, min(150.0, 2000.0 / max(1.0, r)))

def _cm_to_bin10(cm): return int(max(0, min(9, cm // 10)))

# --- NEW: warm-up that wall-follows and auto-labels to CSV ---
async def warmup_autolog(robot: Create3, *, data_csv, seconds: float = WARMUP_SECONDS, dt: float = DT):
    import csv, time
    new_file = not data_csv.exists()
    f = data_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["location", "IR1", "IR5"])

    # PID for wall-follow
    from .pid import PID
    controller = PID(kp=0.4, ki=0.02, kd=0.1, setpoint=SETPOINT_CM)
    await robot.set_lights_on_rgb(0,255,0)
    await robot.set_wheel_speeds(FORWARD, FORWARD)

    # FSM stats for door detection on IR distance
    base = None; var = 25.0
    state = "Wall"; rise_cnt = 0; door_cnt = 0; refractory = 0

    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            pm1 = await robot.get_ir_proximity()
            if pm1 is None or pm1.sensors is None:
                await asyncio.sleep(dt); continue
            d1 = _prox_to_cm(pm1.sensors[RIGHT_IR_IDX])
            await asyncio.sleep(0.02)
            pm2 = await robot.get_ir_proximity()
            if pm2 is None or pm2.sensors is None:
                await asyncio.sleep(dt); continue
            d2 = _prox_to_cm(pm2.sensors[RIGHT_IR_IDX])
            dist_cm = 0.5*(d1+d2)

            # PID control
            u = controller.update(measurement=dist_cm, dt=dt)
            L = max(min(FORWARD + u, MAX_W), MIN_W)
            R = max(min(FORWARD - u, MAX_W), MIN_W)
            await robot.set_wheel_speeds(L, R)

            # EWMA baseline + variance
            if base is None:
                base = dist_cm
            else:
                base = (1 - AUTO_EWMA_ALPHA) * base + AUTO_EWMA_ALPHA * dist_cm
            dev = dist_cm - base
            var = (1 - AUTO_EWVAR_ALPHA) * var + AUTO_EWVAR_ALPHA * (dev*dev)
            sigma = max(3.0, var**0.5)
            rise_th = base + AUTO_K_RISE*sigma
            fall_th = base + AUTO_K_FALL*sigma
            if refractory > 0: refractory -= 1

            # FSM labels while wall-following
            label = "Wall"
            if state == "Wall":
                if refractory == 0 and dist_cm > rise_th:
                    rise_cnt += 1
                    if rise_cnt >= AUTO_MIN_RISE_SAMPLES:
                        state = "Door_Start"; label = "Door_Start"
                        door_cnt = 0; rise_cnt = 0
                        await robot.set_lights_on_rgb(255,255,0)  # yellow
                else:
                    rise_cnt = 0; label = "Wall"
                    await robot.set_lights_on_rgb(0,255,0)      # green
            elif state == "Door_Start":
                state = "Door"; label = "Door"
                door_cnt = 1
                await robot.set_lights_on_rgb(0,0,255)          # blue
            elif state == "Door":
                door_cnt += 1
                if dist_cm <= fall_th and door_cnt >= AUTO_MIN_DOOR_SAMPLES:
                    state = "Door_Passed"; label = "Door_Passed"
                    refractory = AUTO_REFRACTORY_STEPS
                    await robot.set_lights_on_rgb(255,0,255)    # magenta
                else:
                    label = "Door"; await robot.set_lights_on_rgb(0,0,255)
            elif state == "Door_Passed":
                state = "Wall"; label = "Wall"
                await robot.set_lights_on_rgb(0,255,0)

            b = _cm_to_bin10(dist_cm)
            w.writerow([label, b, b]); f.flush()
            print(f"[warmup] base={base:5.1f} σ={sigma:4.1f} d={dist_cm:5.1f} bin={b}  label={label:12s}  L,R=({L:.1f},{R:.1f})")
            await asyncio.sleep(dt)
    finally:
        f.close()
        print(f"[warmup] saved → {data_csv.resolve()}")

async def run_controller(robot: Create3, *, cpts_dir, door_states, dt: float = DT,
                         setpoint_cm: float = SETPOINT_CM,
                         forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    controller = PID(kp=0.4, ki=0.02, kd=0.1, setpoint=setpoint_cm)
    await robot.set_lights_on_rgb(0,255,0)
    await robot.set_wheel_speeds(forward, forward)

    history = []
    try:
        while True:
            pm1 = await robot.get_ir_proximity()
            if pm1 is None or pm1.sensors is None:
                await asyncio.sleep(dt); continue
            d1 = prox_to_cm(pm1.sensors[RIGHT_IR_IDX])
            await asyncio.sleep(0.02)
            pm2 = await robot.get_ir_proximity()
            if pm2 is None or pm2.sensors is None:
                await asyncio.sleep(dt); continue
            d2 = prox_to_cm(pm2.sensors[RIGHT_IR_IDX])
            dist_cm = 0.5*(d1+d2)

            u = controller.update(measurement=dist_cm, dt=dt)
            L = max(min(forward + u, max_w), min_w)
            R = max(min(forward - u, max_w), min_w)
            await robot.set_wheel_speeds(L, R)

            reading = {"IR1": cm_to_bin10(dist_cm), "IR5": cm_to_bin10(dist_cm)}
            b = belief(reading, {"cpts_dir": str(cpts_dir), "door_states": door_states})
            p_bins = b.get("p_distance_bins", {})       # {bin_index: prob}
            expected_cm = sum((bin_idx*10 + 5) * p for bin_idx, p in p_bins.items())
            print(f"[control] E[distance]≈{expected_cm:.1f}cm  p_bins={p_bins}")
            
            history.append(b)
            p_now = b.get("p_door_passed", 0.0)
            p_ago = door_passed_10cm_ago(history, cm_per_step=1.0, door_states=tuple(door_states))

            if p_now > 0.6 or p_ago > 0.6:
                await robot.set_lights_on_rgb(255,0,255)
                await robot.set_wheel_speeds(max(min_w, forward*0.5), max(min_w, forward*0.5))
            else:
                await robot.set_lights_on_rgb(0,255,0)

            print(f"[control] dist≈{dist_cm:5.1f}cm bin={reading['IR1']} door_now={p_now:.2f} door_10cm_ago={p_ago:.2f} L,R=({L:.1f},{R:.1f})")
            await asyncio.sleep(dt)
    finally:
        await robot.set_wheel_speeds(0,0)
        await robot.set_lights_on_rgb(255,0,0)
