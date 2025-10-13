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

async def apply_pid_to_motors(robot: Create3, controller: PID, dist_cm: float, dt: float,
                              wall_side: str, forward: float, max_w: float, min_w: float):
    """Calculates PID output and applies it to the robot's motors."""
    u = controller.update(measurement=dist_cm, dt=dt)
    if wall_side == 'right':
        L = max(min(forward - u * 1, max_w), min_w)
        R = max(min(forward + u * 1, max_w), min_w)
    else:  # left
        L = max(min(forward + u * 1, max_w), min_w)
        R = max(min(forward - u * 1, max_w), min_w)
    await robot.set_wheel_speeds(L, R)
    return L, R

async def initialize_pid_wall_follower(robot: Create3, setpoint_cm: float, forward: float) -> PID:
    """Initializes PID controller and sets initial robot lights and speed."""
    controller = PID(kp=0.4, ki=0.02, kd=0.1, setpoint=setpoint_cm)
    await robot.set_lights_on_rgb(0, 255, 0)
    await robot.set_wheel_speeds(forward, forward)
    return controller

def initialize_history_deques(history_len: int = 9) -> tuple:
    """Initializes and returns a tuple of history deques."""
    ir_history = deque([0]*history_len, maxlen=history_len)
    pid_p_history = deque([0.0]*history_len, maxlen=history_len)
    pid_i_history = deque([0.0]*history_len, maxlen=history_len)
    pid_d_history = deque([0.0]*history_len, maxlen=history_len)
    bumper_history = deque([False]*history_len, maxlen=history_len)
    return ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history

async def get_label():
    """Asynchronously and non-blockingly waits for a keypress from the user."""
    label_map = {"w": "Wall", "s": "Door_Start", "d": "Door", "p": "Door_Passed"}

    print("Please enter a label for the current location: (w)all, (s)tart, (d)oor, (p)assed:")
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
        
        await asyncio.sleep(0.05) 

    label = label_map.get(label_shortcut, "Wall")
    print(f"  Labelled as: {label}")
    return label

def update_histories(
    ir_history: deque,
    pid_p_history: deque,
    pid_i_history: deque,
    pid_d_history: deque,
    bumper_history: deque,
    dist_cm: float,
    controller: PID,
    bumpers,
    dt: float,
    setpoint_cm: float
):
    """Updates all history deques with new sensor data."""
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
    bumper_history.append(any(bumpers))

async def collect_data_manual(robot: Create3, *, data_csv, dt: float = DT,
                               setpoint_cm: float = SETPOINT_CM,
                               forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and collects time-shifted data with manual annotation.
    """
    controller = await initialize_pid_wall_follower(robot, setpoint_cm, forward)
    ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history = initialize_history_deques()

    # CSV setup
    new_file = not data_csv.exists()
    f = data_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["location"] + FEATURES)

    label = None

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
            
            L, R = await apply_pid_to_motors(robot, controller, dist_cm, dt,
                                             WALL_SIDE, forward, max_w, min_w)

            # Manual annotation
            distance_since_last_prompt += (((pos["x"] - last_pos["x"])**2 + (pos["y"] - last_pos["y"])**2)**0.5)
            
            if distance_since_last_prompt >= 10:
                await robot.set_wheel_speeds(0, 0)
                if label is not None:
                    label = await get_label()
                    # Write to CSV
                    row = [label] + list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history)
                    w.writerow(row)
                    f.flush()
                else:
                    label = await get_label()

                update_histories(ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history,
                dist_cm, controller, bumpers, dt, setpoint_cm)

                distance_since_last_prompt = 0

            print(f"[collect] dist≈{dist_cm:5.1f}cm bin={ir_history[-1]} L,R=({L:.1f},{R:.1f}) distance_since_last_prompt = {distance_since_last_prompt:5.1f}")
            #print (f"pos.x = {pos["x"]}, pos.y = {pos["y"]},  last_pos.x = {last_pos["x"]}, last_pos.y = {last_pos["y"]}")
            last_pos = pos.copy()
            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        f.close()
        print(f"[collect] saved → {data_csv.resolve()}")

async def run_location_predictor(robot: Create3, *, cpts_dir, door_states, dt: float = DT,
                                 setpoint_cm: float = SETPOINT_CM,
                                 forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and predicts location every 10cm, saving predictions to a CSV file.
    """
    controller = await initialize_pid_wall_follower(robot, setpoint_cm, forward)
    ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history = initialize_history_deques()
    
    belief_history = []
    distance_since_last_prediction = 0
    pose = await robot.get_position()
    last_pos = np.array((pose.x, pose.y), dtype=pose_dtype)

    # --- CSV setup for predictions ---
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    data_dir = cpts_dir.parent / "data"
    predictions_csv_path = data_dir / f"predictions_{timestamp_str}.csv"
    
    pred_f = predictions_csv_path.open("w", newline="", encoding="utf-8")
    pred_w = csv.writer(pred_f)
    header = ["Timestamp", "Predicted_Location", "Door_Passed_10cm_Ago", "Predicted_Distance_cm"]
    pred_w.writerow(header)
    pred_f.flush()

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
            
            await apply_pid_to_motors(robot, controller, dist_cm, dt,
                                      WALL_SIDE, forward, max_w, min_w)

            update_histories(ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history,
                             dist_cm, controller, bumpers, dt, setpoint_cm)

            # Create readings dictionary from history for belief calculation
            all_history = list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history)
            readings = {feature: value for feature, value in zip(FEATURES, all_history)}
            b = belief(readings, {"cpts_dir": str(cpts_dir), "door_states": door_states})
            belief_history.append(b)

            distance_since_last_prediction += (((pos["x"] - last_pos["x"])**2 + (pos["y"] - last_pos["y"])**2)**0.5)
            last_pos = pos.copy()

            if distance_since_last_prediction >= 10:
                print("\n--- PREDICTIONS ---")
                
                # --- Initialize variables for this prediction cycle ---
                predicted_location = "N/A"
                p_ago = 0.0
                expected_cm = 0.0

                # Location prediction
                posterior = b.get("posterior", {})
                if posterior:
                    predicted_location = max(posterior, key=posterior.get)
                    print(f"Predicted Location: {predicted_location}")
                    print("  Location Probabilities:")
                    for location, probability in sorted(posterior.items()):
                        print(f"    - {location}: {probability:.4f}")
                else:
                    print("Could not calculate location belief.")

                # Door passed 10cm ago prediction
                p_ago = door_passed_10cm_ago(belief_history, cm_per_step=1.0, door_states=tuple(door_states))
                print(f"\nProbability of having passed a door 10cm ago: {p_ago:.4f}")
                door_passed_decision = "NO"
                if p_ago > 0.6:
                    door_passed_decision = "YES"
                    print("  DECISION: YES, a door was passed ~10cm ago.")
                else:
                    print("  DECISION: NO, a door was not passed ~10cm ago.")

                # Distance from wall prediction
                p_distance_bins = b.get("p_distance_bins", {})
                if p_distance_bins:
                    expected_cm = sum((bin_idx*10 + 5) * p for bin_idx, p in p_distance_bins.items())
                    print(f"\nPredicted Distance from Wall: {expected_cm:.1f} cm")
                    print("  Distance Distribution:")
                    for bin_idx, prob in sorted(p_distance_bins.items()):
                        if prob > 0.01: # Only print significant probabilities
                            print(f"    - {bin_idx*10}-{(bin_idx+1)*10} cm: {prob:.4f}")
                else:
                    print("Could not calculate distance distribution.")
                
                print("-------------------\n")

                # --- Write to CSV ---
                pred_w.writerow([time.time(), predicted_location, door_passed_decision, expected_cm])
                pred_f.flush()

                distance_since_last_prediction = 0

            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        pred_f.close()
        print(f"\n[predictor] Predictions saved to {predictions_csv_path.resolve()}")
