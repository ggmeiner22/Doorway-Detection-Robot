from __future__ import annotations
import asyncio, csv, time
import numpy as np
from collections import deque
from irobot_edu_sdk.robots import Create3
from .pid import PID
from .robot_io import prox_to_cm, cm_to_bin12, discretize_p_10_bins, discretize_i_10_bins, discretize_d_10_bins
from .belief_network import belief, door_passed_10cm_ago
from .config import (
    DT, SETPOINT_CM, FORWARD, MAX_W, MIN_W, IR_SENSOR_IDX, WALL_SIDE,
    DOOR_STATES, CPTS_DIR, FEATURES, WARMUP_SECONDS,
    AUTO_K_RISE, AUTO_K_FALL, AUTO_MIN_RISE_SAMPLES, AUTO_MIN_DOOR_SAMPLES,
    AUTO_REFRACTORY_STEPS, AUTO_EWMA_ALPHA, AUTO_EWVAR_ALPHA
)

import keyboard
pose_dtype = np.dtype([("x", np.float64), ("y", np.float64)])
async def collect_data_manual(robot: Create3, *, data_csv, dt: float = DT,
                               setpoint_cm: float = SETPOINT_CM,
                               forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and collects time-shifted data with manual annotation.
    """
    # PID for wall-follow
    controller = PID(kp=0.4, ki=0.02, kd=0.1, setpoint=setpoint_cm)
    await robot.set_lights_on_rgb(0, 255, 0)
    await robot.set_wheel_speeds(forward, forward)

    # History deques
    history_len = 9
    ir_history = deque([0]*history_len, maxlen=history_len)
    pid_p_history = deque([0.0]*history_len, maxlen=history_len)
    pid_i_history = deque([0.0]*history_len, maxlen=history_len)
    pid_d_history = deque([0.0]*history_len, maxlen=history_len)
    bumper_history = deque([False]*history_len, maxlen=history_len)

    # CSV setup
    new_file = not data_csv.exists()
    f = data_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["location"] + FEATURES)

    distance_since_last_prompt = 0
    pose = await robot.get_position()
    last_pos = np.array((pose.x, pose.y), dtype=pose_dtype)

    try:
        while not keyboard.is_pressed('q'):
            # Get sensor data
            pm1 = await robot.get_ir_proximity()
            await asyncio.sleep(0.05)
            pm2 = await robot.get_ir_proximity()
            bumpers = await robot.get_bumpers()
            pose = await robot.get_position()
            pos = np.array((pose.x, pose.y), dtype=pose_dtype)

            if pm1 is None or pm1.sensors is None or pm2 is None or pm2.sensors is None or pos is None or last_pos is None:
                await asyncio.sleep(dt)
                continue

            d1 = prox_to_cm(pm1.sensors[IR_SENSOR_IDX])
            d2 = prox_to_cm(pm2.sensors[IR_SENSOR_IDX])
            dist_cm = 0.5*(d1+d2)
            
            u = controller.update(measurement=dist_cm, dt=dt)
            if WALL_SIDE == 'right':
                L = max(min(forward - u * 1, max_w), min_w)
                R = max(min(forward + u * 1, max_w), min_w)
            else:  # left
                L = max(min(forward + u * 1, max_w), min_w)
                R = max(min(forward - u * 1, max_w), min_w)
            await robot.set_wheel_speeds(L, R)

            # Update history
            ir_history.append(cm_to_bin12(dist_cm))
            # P term
            p_binned = discretize_p_10_bins(dist_cm, setpoint_cm)
            pid_p_history.append(p_binned)
            # I term
            i_val = controller._integ
            i_binned = discretize_i_10_bins(i_val, setpoint_cm)
            pid_i_history.append(i_binned)
            # D term
            d_val = 0.0
            if controller._prev is not None and controller._prev_prev is not None:
                d_val = (controller._prev - controller._prev_prev) / dt
            d_binned = discretize_d_10_bins(d_val, setpoint_cm)
            pid_d_history.append(d_binned)
            bumper_history.append(any([bumpers.left, bumpers.right, bumpers.front_left, bumpers.front_right]))

            # Manual annotation
            distance_since_last_prompt += (((pos["x"] - last_pos["x"])**2 + (pos["y"] - last_pos["y"])**2)**0.5)
            

            label = "Wall"
            label_map = {"w": "Wall", "s": "Door_Start", "d": "Door", "p": "Door_Passed"}
            if distance_since_last_prompt >= 10:
                await robot.set_wheel_speeds(0, 0)
                print("Please enter a label for the wcurrent location: (w)all, (s)tart, (d)oor, (p)assed:")
                label_shortcut = None
                while label_shortcut is None:
                    if keyboard.is_pressed('w'):
                        label_shortcut = 'w'
                    elif keyboard.is_pressed('s'):
                        label_shortcut = 's'
                    elif keyboard.is_pressed('d'):
                        label_shortcut = 'd'
                    elif keyboard.is_pressed('p'):
                        label_shortcut = 'p'
                    await asyncio.sleep(0.05) # prevent busy-waiting

                label = label_map.get(label_shortcut, "Wall")
                print(f"  Labelled as: {label}")
                distance_since_last_prompt = 0

            # Write to CSV
            row = [label] + list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history)
            w.writerow(row)
            f.flush()

            print(f"[collect] dist≈{dist_cm:5.1f}cm bin={ir_history[-1]} L,R=({L:.1f},{R:.1f}) distance_since_last_prompt = {distance_since_last_prompt:5.1f}")
            #print (f"pos.x = {pos["x"]}, pos.y = {pos["y"]},  last_pos.x = {last_pos["x"]}, last_pos.y = {last_pos["y"]}")
            last_pos = pos.copy()
            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        f.close()
        print(f"[collect] saved → {data_csv.resolve()}")

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
            d1 = prox_to_cm(pm1.sensors[IR_SENSOR_IDX])
            await asyncio.sleep(0.02)
            pm2 = await robot.get_ir_proximity()
            if pm2 is None or pm2.sensors is None:
                await asyncio.sleep(dt); continue
            d2 = prox_to_cm(pm2.sensors[IR_SENSOR_IDX])
            dist_cm = 0.5*(d1+d2)

            u = controller.update(measurement=dist_cm, dt=dt)
            if WALL_SIDE == 'right':
                L = max(min(forward - u * 1, max_w), min_w)
                R = max(min(forward + u * 1, max_w), min_w)
            else:  # left
                L = max(min(forward + u * 1, max_w), min_w)
                R = max(min(forward - u * 1, max_w), min_w)
            await robot.set_wheel_speeds(L, R)

            reading = {"IR1": cm_to_bin12(dist_cm), "IR5": cm_to_bin12(dist_cm)}
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

async def run_location_predictor(robot: Create3, *, cpts_dir, dt: float = DT,
                                 setpoint_cm: float = SETPOINT_CM,
                                 forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and predicts location every 10cm.
    """
    # PID for wall-follow
    controller = PID(kp=0.4, ki=0.02, kd=0.1, setpoint=setpoint_cm)
    await robot.set_lights_on_rgb(0, 255, 0)
    await robot.set_wheel_speeds(forward, forward)

    # History deques
    history_len = 9
    ir_history = deque([0]*history_len, maxlen=history_len)
    pid_p_history = deque([0.0]*history_len, maxlen=history_len)
    pid_i_history = deque([0.0]*history_len, maxlen=history_len)
    pid_d_history = deque([0.0]*history_len, maxlen=history_len)
    bumper_history = deque([False]*history_len, maxlen=history_len)

    distance_since_last_prediction = 0
    pose = await robot.get_position()
    last_pos = np.array((pose.x, pose.y), dtype=pose_dtype)

    try:
        while True:
            # Get sensor data
            pm1 = await robot.get_ir_proximity()
            await asyncio.sleep(0.05)
            pm2 = await robot.get_ir_proximity()
            bumpers = await robot.get_bumpers()
            pose = await robot.get_position()
            pos = np.array((pose.x, pose.y), dtype=pose_dtype)

            if pm1 is None or pm1.sensors is None or pm2 is None or pm2.sensors is None or pos is None or last_pos is None:
                await asyncio.sleep(dt)
                continue

            d1 = prox_to_cm(pm1.sensors[IR_SENSOR_IDX])
            d2 = prox_to_cm(pm2.sensors[IR_SENSOR_IDX])
            dist_cm = 0.5*(d1+d2)
            
            u = controller.update(measurement=dist_cm, dt=dt)
            if WALL_SIDE == 'right':
                L = max(min(forward - u * 1, max_w), min_w)
                R = max(min(forward + u * 1, max_w), min_w)
            else:  # left
                L = max(min(forward + u * 1, max_w), min_w)
                R = max(min(forward - u * 1, max_w), min_w)
            await robot.set_wheel_speeds(L, R)

            # Update history
            ir_history.append(cm_to_bin12(dist_cm))
            p_binned = discretize_p_10_bins(dist_cm, setpoint_cm)
            pid_p_history.append(p_binned)
            i_val = controller._integ
            i_binned = discretize_i_10_bins(i_val, setpoint_cm)
            pid_i_history.append(i_binned)
            d_val = 0.0
            if controller._prev is not None and controller._prev_prev is not None:
                d_val = (controller._prev - controller._prev_prev) / dt
            d_binned = discretize_d_10_bins(d_val, setpoint_cm)
            pid_d_history.append(d_binned)
            bumper_history.append(any([bumpers.left, bumpers.right, bumpers.front_left, bumpers.front_right]))

            distance_since_last_prediction += (((pos["x"] - last_pos["x"])**2 + (pos["y"] - last_pos["y"])**2)**0.5) * 100 # meters to cm
            last_pos = pos.copy()

            if distance_since_last_prediction >= 10:
                print("--- Predicting location ---")
                
                # Create readings dictionary from history
                all_history = list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history)
                readings = {feature: value for feature, value in zip(FEATURES, all_history)}

                # Get belief
                b = belief(readings, {"cpts_dir": str(cpts_dir)})
                posterior = b.get("posterior", {})

                if posterior:
                    # Find predicted location
                    predicted_location = max(posterior, key=posterior.get)
                    print(f"Predicted Location: {predicted_location}")

                    # Print normalized probabilities
                    print("Location Probabilities:")
                    for location, probability in sorted(posterior.items()):
                        print(f"  - {location}: {probability:.4f}")
                else:
                    print("Could not calculate belief.")

                distance_since_last_prediction = 0

            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)