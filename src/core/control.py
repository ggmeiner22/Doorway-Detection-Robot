from __future__ import annotations
import logging, sys
from pathlib import Path
import asyncio, csv, time
import numpy as np
from collections import deque
from irobot_edu_sdk.robots import Create3
from irobot_edu_sdk.music import Note
from .pid import PID
from .robot_io import prox_to_cm, cm_to_bin12, discretize_p_10_bins, discretize_i_10_bins, discretize_d_10_bins, discretize_odo
from .belief_network import belief
from .pomdp_belief_network import pomdp_belief, get_expected_reward
from .config import (
    DT, SETPOINT_CM, FORWARD, MAX_W, MIN_W, IR_SENSOR_IDX, WALL_SIDE,
    FEATURES, POMDP_FEATURES, STOP, YAW_DEG_PER_SEC, RIGHT_IR_IDX, LEFT_IR_IDX,  CPTS_DIR,
    HOME_RADIUS, DECEL_RADIUS
)

import keyboard
pose_dtype = np.dtype([("x", np.float64), ("y", np.float64)])


def make_logger(name: str, log_dir: Path, *, truncate: bool = True) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}.log"

    # 'w' = clear on each run; 'a' = append across runs
    mode = 'w' if truncate else 'a'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_path, mode=mode, encoding="utf-8", delay=True)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

logs_dir = CPTS_DIR.parent / "logs"
logger = make_logger("predictor", logs_dir)
logger.info(f"Logging to {logs_dir}")

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
    pid_p_history = deque([0]*history_len, maxlen=history_len)
    pid_i_history = deque([0]*history_len, maxlen=history_len)
    pid_d_history = deque([0]*history_len, maxlen=history_len)
    bumper_history = deque([False]*history_len, maxlen=history_len)
    return ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history


async def get_label():
    """Asynchronously and non-blockingly waits for a keypress from the user."""
    # Updated to expanded labels
    # 0=Wall_0, 1=Door_Start_1, 2=Door_1, 3=Door_Passed_1
    # 4=Wall_1, 5=Door_Start_2, 6=Door_2, 7=Door_Passed_2
    # 8=Wall_2, 9=Door_Start_3, a=Door_3, b=Door_Passed_3
    # e=Wall_End
    # Simplified map for user convenience (user must know where they are)
    
    logger.info("Enter label: (w)all, (s)tart, (d)oor, (p)assed, (e)nd. \nPress (r) for REWARD!")
    
    label_input = None
    reward = 0
    
    while label_input is None:
        # Reward toggle
        if keyboard.is_pressed('r'):
            reward = 1
            logger.info("  >> REWARD FLAGGED! <<")
            # Debounce
            await asyncio.sleep(0.3)

        # Locations
        if keyboard.is_pressed('w'):
            label_input = 'Wall'
            pass 
        pass
    return "Wall" # Placeholder


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


async def wall_follow_step(robot: Create3, controller: PID, wall_side: str, forward: float, max_w: float, min_w: float, dt: float):
    """A single step of wall following, to be used by both collect and run."""
    pm1 = await robot.get_ir_proximity()
    await asyncio.sleep(0.05)
    pm2 = await robot.get_ir_proximity()
    bumpers = await robot.get_bumpers()

    if any(bumpers):
        left_hit, right_hit = bumpers
        logger.info(f"Bumper hit: left={left_hit}, right={right_hit}, wall_side: {wall_side}. Backing off.")
        await robot.set_wheel_speeds(0,0)
        await robot.move(-10)
        if wall_side == 'right':
            await robot.turn_left(25)
        else: # 'left'
            await robot.turn_right(25)
        # After backing off, reset controller and histories to avoid PID wind-up from the bump
        controller.reset()
        # ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history = initialize_history_deques()
        logger.info("Resuming wall following.")
        return None, None, None

    pose = await robot.get_position()

    if pm1 is None or pm1.sensors is None or pm2 is None or pm2.sensors is None or pose is None:
        await asyncio.sleep(dt)
        return None, None, None

    IR_SENSOR_IDX = RIGHT_IR_IDX if wall_side == 'right' else LEFT_IR_IDX
    d1 = prox_to_cm(pm1.sensors[IR_SENSOR_IDX])
    d2 = prox_to_cm(pm2.sensors[IR_SENSOR_IDX])
    dist_cm = 0.5*(d1+d2)

    await apply_pid_to_motors(robot, controller, dist_cm, dt,
                              wall_side, forward, max_w, min_w)

    return dist_cm, bumpers, pose


async def collect_data_manual(robot: Create3, *, data_csv, dt: float = DT,
                               setpoint_cm: float = SETPOINT_CM,
                               forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and collects time-shifted data with manual annotation.
    """
    controller = await initialize_pid_wall_follower(robot, setpoint_cm, forward)
    ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history = initialize_history_deques()
    wall_side = WALL_SIDE # Localize the var

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
            dist_cm, bumpers, pose = await wall_follow_step(robot, controller, wall_side, forward, max_w, min_w, dt)
            if dist_cm is None:
                continue

            pos = np.array((pose.x, pose.y), dtype=pose_dtype)

            # Manual annotation
            distance_since_last_prompt += (((pos["x"] - last_pos["x"])**2 + (pos["y"] - last_pos["y"])**2)**0.5)

            if distance_since_last_prompt >= 10:
                await robot.set_wheel_speeds(0, 0)
                
                # Simple label input for standard collection
                logger.info("Label? (w)all, (s)tart, (d)oor, (p)assed")
                while True:
                    if keyboard.is_pressed('w'): label="Wall"; break
                    if keyboard.is_pressed('s'): label="Door_Start"; break
                    if keyboard.is_pressed('d'): label="Door"; break
                    if keyboard.is_pressed('p'): label="Door_Passed"; break
                    await asyncio.sleep(0.05)
                
                # Write to CSV
                row = [label] + list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history)
                w.writerow(row)
                f.flush()
                logger.info(f"Saved: {label}")

                distance_since_last_prompt = 0

            update_histories(ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history,
            dist_cm, controller, bumpers, dt, setpoint_cm)

            logger.info(f"[collect] dist≈{dist_cm:5.1f}cm bin={ir_history[-1]} distance_since_last_prompt = {distance_since_last_prompt:5.1f}")
            last_pos = pos.copy()
            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        f.close()
        logger.info(f"[collect] saved → {data_csv.resolve()}")


async def run_location_predictor(robot: Create3, *, cpts_dir, door_states, belief_func=belief, dt: float = DT, return_home: bool = False,
                                 setpoint_cm: float = SETPOINT_CM,
                                 forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and predicts location every 10cm, saving predictions to a CSV file.
    """
    controller = await initialize_pid_wall_follower(robot, setpoint_cm, forward)
    ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history = initialize_history_deques()
    wall_side = WALL_SIDE # Localize the variable

    last_belief = None
    distance_since_last_prediction = 0
    pose = await robot.get_position()
    last_pos = np.array((pose.x, pose.y), dtype=pose_dtype)

    # remember where we started
    start_pose = await robot.get_position()
    start_xy = np.array((start_pose.x, start_pose.y), dtype=pose_dtype)
    returning = False     # set True after the 180° turn


    # --- CSV setup for predictions ---
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    data_dir = cpts_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv_path = data_dir / f"predictions_{timestamp_str}.csv"

    pred_f = predictions_csv_path.open("w", newline="", encoding="utf-8")
    pred_w = csv.writer(pred_f)
    header = ["Timestamp", "Predicted_Location", "Door_Passed_10cm_Ago", "Predicted_Distance_cm", "Doors_Passed"]
    pred_w.writerow(header)
    pred_f.flush()

    num_of_doors_passed = 0  # Used if return_home is True only

    try:
        while True:
            dist_cm, bumpers, pose = await wall_follow_step(robot, controller, wall_side, forward, max_w, min_w, dt)
            if dist_cm is None:
                continue

            pos = np.array((pose.x, pose.y), dtype=pose_dtype)

            # --- homing control (only after we've turned around) ---
            speed_scale = 1.0
            if returning:
                # distance from current position to start
                dist_home = float(np.hypot(pos["x"] - start_xy["x"], pos["y"] - start_xy["y"]))

                # Gentle decel as we approach home
                if dist_home < DECEL_RADIUS:
                    # scale forward speed (down to ~30% near HOME_RADIUS)
                    speed_scale = max(0.3, min(1.0, dist_home / DECEL_RADIUS))

                # if we’re “home”, stop and exit
                if dist_home <= HOME_RADIUS:
                    logger.info(f"[predictor] Reached home (≈{dist_home:.1f} units). Stopping.")
                    await robot.set_wheel_speeds(STOP, STOP)
                    await asyncio.sleep(1)
                    break

            update_histories(ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history,
                                dist_cm, controller, bumpers, dt, setpoint_cm)
            # Create readings dictionary from history for belief calculation
            all_history = list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history)
            readings = {feature: value for feature, value in zip(FEATURES, all_history)}
            
            import inspect
            sig = inspect.signature(belief_func)
            if 'last_belief' in sig.parameters:
                b = belief_func(readings, last_belief['posterior'] if last_belief else None, {"cpts_dir": str(cpts_dir), "door_states": door_states})
            else:
                b = belief_func(readings, {"cpts_dir": str(cpts_dir), "door_states": door_states})

            distance_since_last_prediction += (((pos["x"] - last_pos["x"])**2 + (pos["y"] - last_pos["y"])**2)**0.5)
            last_pos = pos.copy()

            if distance_since_last_prediction >= 10:
                await robot.set_wheel_speeds(0, 0)
                await asyncio.sleep(1)
                logger.info("\n--- PREDICTIONS ---")

                # --- Initialize variables for this prediction cycle ---
                predicted_location = "N/A"
                p_ago = 0.0
                expected_cm = 0.0

                # Location prediction
                posterior = b.get("posterior", {})
                if posterior:
                    predicted_location = max(posterior, key=posterior.get)
                    logger.info(f"Predicted Location: {predicted_location}")
                    logger.info("  Location Probabilities:")
                    for location, probability in sorted(posterior.items()):
                        logger.info(f"    - {location}: {probability:.4f}")
                else:
                    logger.info("Could not calculate location belief.")

                # Door passed 10cm ago prediction
                if last_belief:
                    p_ago = sum(last_belief["posterior"].get(s, 0.0) for s in door_states)
                door_passed_decision = "NO"
                if p_ago > 0.6:
                    door_passed_decision = "YES"
                    num_of_doors_passed += 1
                
                logger.info(f"Doors passed so far: {num_of_doors_passed}")

                # Distance from wall prediction
                p_distance_bins = b.get("p_distance_bins", {})
                if p_distance_bins:
                    expected_cm = sum((bin_idx + 0.5) * p for bin_idx, p in p_distance_bins.items())
                    logger.info(f"\nPredicted Distance from Wall: {expected_cm:.1f} cm")
                    logger.info("  Distance Distribution (0-12 cm):")
                    for bin_idx, prob in sorted(p_distance_bins.items()):
                        logger.info(f"    - {bin_idx}-{bin_idx + 1} cm: {prob:.4f}")
                else:
                    logger.info("Could not calculate distance distribution.")

                logger.info("-------------------\n")

                # --- Write to CSV ---
                pred_w.writerow([time.time(), predicted_location, door_passed_decision, expected_cm, num_of_doors_passed])
                pred_f.flush()

                last_belief = b
                distance_since_last_prediction = 0

                # If we are turning back to come home
                if return_home and num_of_doors_passed == 3:
                    # Stop robot
                    await robot.set_wheel_speeds(STOP, STOP)
                    await asyncio.sleep(2)

                    ## Turn robot around
                    await turn_around_create3(robot, angle_deg=180, direction="left")

                    ## Swap which wall we follow
                    wall_side = "left" if wall_side == "right" else "right"
                    logger.info(f"[predictor] Turned 180°, now following {wall_side} wall")
                    returning = True
                    return_home = False

            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        pred_f.close()
        logger.info(f"\n[predictor] Predictions saved to {predictions_csv_path.resolve()}")


def initialize_history_deques_pomdp(history_len: int = 9) -> tuple:
    """Initializes and returns a tuple of history deques for POMDP."""
    ir_history = deque([0]*history_len, maxlen=history_len)
    pid_p_history = deque([0]*history_len, maxlen=history_len)
    pid_i_history = deque([0]*history_len, maxlen=history_len)
    pid_d_history = deque([0]*history_len, maxlen=history_len)
    bumper_history = deque([False]*history_len, maxlen=history_len)
    odo_history = deque([0]*history_len, maxlen=history_len)
    return ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history, odo_history


def update_histories_pomdp(
    ir_history: deque,
    pid_p_history: deque,
    pid_i_history: deque,
    pid_d_history: deque,
    bumper_history: deque,
    odo_history: deque,
    dist_cm: float,
    controller: PID,
    bumpers,
    dt: float,
    setpoint_cm: float,
    delta_dist: float
):
    """Updates all history deques with new sensor data for POMDP."""
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
    odo_history.append(discretize_odo(delta_dist))


async def collect_data_pomdp(robot: Create3, *, data_csv, dt: float = DT,
                               setpoint_cm: float = SETPOINT_CM,
                               forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and collects time-shifted data with manual annotation for POMDP.
    Includes REWARD collection.
    """
    controller = await initialize_pid_wall_follower(robot, setpoint_cm, forward)
    ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history, odo_history = initialize_history_deques_pomdp()
    wall_side = WALL_SIDE # Localize the var

    # CSV setup
    new_file = not data_csv.exists()
    f = data_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["location"] + POMDP_FEATURES)

    label = None
    current_door_idx = 0 # 0 = before door 1

    distance_since_last_prompt = 0
    pose = await robot.get_position()
    last_pos = np.array((pose.x, pose.y), dtype=pose_dtype)

    try:
        while not keyboard.is_pressed('q'):
            dist_cm, bumpers, pose = await wall_follow_step(robot, controller, wall_side, forward, max_w, min_w, dt)
            if dist_cm is None:
                continue

            pos = np.array((pose.x, pose.y), dtype=pose_dtype)
            delta_dist = float(np.hypot(pos["x"] - last_pos["x"], pos["y"] - last_pos["y"]))

            # Manual annotation
            distance_since_last_prompt += delta_dist

            if distance_since_last_prompt >= 10:
                await robot.set_wheel_speeds(0, 0)
                
                # Extended labeling for POMDP
                logger.info(f"Current Door Context: {current_door_idx+1} (0-based={current_door_idx})")
                logger.info("Label? (w)all, (s)tart, (d)oor, (p)assed, (e)nd. \n[n]ext door idx, [r]eward")
                
                reward = 0
                last_print_time = time.time()
                
                while True:
                    # Periodic status update
                    if time.time() - last_print_time > 2.0:
                        print(f"Waiting... Context: {current_door_idx+1} | Reward: {'YES' if reward else 'NO'}")
                        last_print_time = time.time()

                    if keyboard.is_pressed('r'):
                        reward = 1
                        logger.info("REWARD!")
                        print("!!! REWARD MARKED !!!")
                        # Audio feedback
                        try: await robot.play_note(Note.C6, 0.2)
                        except: pass
                        await asyncio.sleep(0.3)
                        last_print_time = time.time()
                    
                    if keyboard.is_pressed('n'):
                        current_door_idx = min(current_door_idx + 1, 3) # Max index 3 -> Wall_End (Absorbing)
                        logger.info(f"Next door context: {current_door_idx+1}")
                        print(f"*** CONTEXT UPDATED TO: {current_door_idx+1} ***")
                        # Audio feedback
                        try: await robot.play_note(Note.A5, 0.2)
                        except: pass
                        await asyncio.sleep(0.3)
                        last_print_time = time.time()

                    # Auto-label if in absorbing state
                    if current_door_idx >= 3:
                        label = "Wall_End"
                        if keyboard.is_pressed('w') or keyboard.is_pressed('s') or keyboard.is_pressed('d') or keyboard.is_pressed('p') or keyboard.is_pressed('e'):
                            break
                    
                    else:
                        if keyboard.is_pressed('w'): 
                            label=f"Wall_{current_door_idx}"
                            break
                        if keyboard.is_pressed('s'): 
                            label=f"Door_Start_{current_door_idx+1}"
                            break
                        if keyboard.is_pressed('d'): 
                            label=f"Door_{current_door_idx+1}"
                            break
                        if keyboard.is_pressed('p'): 
                            label=f"Door_Passed_{current_door_idx+1}"
                            break
                        if keyboard.is_pressed('e'):
                            label = "Wall_End" # Premature end?
                            break
                        
                    await asyncio.sleep(0.001) # Very fast polling
                
                # Write to CSV
                row = [label] + list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history) + list(odo_history) + [reward]
                w.writerow(row)
                f.flush()
                logger.info(f"Saved: {label}, Reward={reward}")

                # Update history ONLY every 10cm (spatial step)
                update_histories_pomdp(ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history, odo_history,
                dist_cm, controller, bumpers, dt, setpoint_cm, distance_since_last_prompt)

                distance_since_last_prompt = 0

            #logger.info(f"[collect] dist≈{dist_cm:5.1f}cm")
            last_pos = pos.copy()
            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        f.close()
        logger.info(f"[collect] saved → {data_csv.resolve()}")


async def run_location_predictor_pomdp(robot: Create3, *, cpts_dir, door_states, belief_func, dt: float = DT, return_home: bool = False,
                                 setpoint_cm: float = SETPOINT_CM,
                                 forward: float = FORWARD, max_w: float = MAX_W, min_w: float = MIN_W):
    """
    Wall-follows, and predicts location every 10cm, saving predictions to a CSV file.
    """
    controller = await initialize_pid_wall_follower(robot, setpoint_cm, forward)
    ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history, odo_history = initialize_history_deques_pomdp()
    wall_side = WALL_SIDE # Localize the variable

    last_belief = None
    distance_since_last_prediction = 0
    pose = await robot.get_position()
    last_pos = np.array((pose.x, pose.y), dtype=pose_dtype)

    # remember where we started
    start_pose = await robot.get_position()
    start_xy = np.array((start_pose.x, start_pose.y), dtype=pose_dtype)
    returning = False     # set True after the 180° turn


    # --- CSV setup for predictions ---
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    data_dir = cpts_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv_path = data_dir / f"predictions_{timestamp_str}.csv"

    pred_f = predictions_csv_path.open("w", newline="", encoding="utf-8")
    pred_w = csv.writer(pred_f)
    header = ["Timestamp", "Predicted_Location", "Door_Passed_10cm_Ago", "Predicted_Distance_cm", "Doors_Passed", "Expected_Reward"]
    pred_w.writerow(header)
    pred_f.flush()

    try:
        while True:
            dist_cm, bumpers, pose = await wall_follow_step(robot, controller, wall_side, forward, max_w, min_w, dt)
            if dist_cm is None:
                continue

            pos = np.array((pose.x, pose.y), dtype=pose_dtype)
            delta_dist = float(np.hypot(pos["x"] - last_pos["x"], pos["y"] - last_pos["y"]))

            # --- homing control (only after we've turned around) ---
            speed_scale = 1.0
            if returning:
                # distance from current position to start
                dist_home = float(np.hypot(pos["x"] - start_xy["x"], pos["y"] - start_xy["y"]))

                # Gentle decel as we approach home
                if dist_home < DECEL_RADIUS:
                    # scale forward speed (down to ~30% near HOME_RADIUS)
                    speed_scale = max(0.3, min(1.0, dist_home / DECEL_RADIUS))

                # if we’re “home”, stop and exit
                if dist_home <= HOME_RADIUS:
                    logger.info(f"[predictor] Reached home (≈{dist_home:.1f} units). Stopping.")
                    await robot.set_wheel_speeds(STOP, STOP)
                    await asyncio.sleep(1)
                    break

            # Accumulate distance
            distance_since_last_prediction += delta_dist
            last_pos = pos.copy()

            if distance_since_last_prediction >= 10:
                if not returning:
                    await robot.set_wheel_speeds(0, 0)
                    await asyncio.sleep(1)
                
                # Update sensor history ONCE per 10cm step
                # We pass the accumulated distance (approx 10cm) as the delta_dist for Odometer binning
                update_histories_pomdp(ir_history, pid_p_history, pid_i_history, pid_d_history, bumper_history, odo_history,
                                    dist_cm, controller, bumpers, dt, setpoint_cm, distance_since_last_prediction)
                
                # Create readings dictionary from history for belief calculation
                all_history = list(ir_history) + list(pid_p_history) + list(pid_d_history) + list(pid_i_history) + list(bumper_history) + list(odo_history)
                
                # NOTE: We do NOT pass 'Reward' in readings here, so belief is updated by sensors only.
                readings = {feature: value for feature, value in zip(POMDP_FEATURES, all_history) if feature != "Reward"}
                
                b = belief_func(readings, last_belief['posterior'] if last_belief else None, {"cpts_dir": str(cpts_dir), "door_states": door_states})
                last_belief = b # Update state

                logger.info("\n--- PREDICTIONS ---")

                # --- Initialize variables for this prediction cycle ---
                predicted_location = "N/A"
                expected_cm = 0.0

                # Location prediction
                posterior = b.get("posterior", {})
                if posterior:
                    predicted_location = max(posterior, key=posterior.get)
                    logger.info(f"Predicted Location: {predicted_location}")
                    logger.info("  Location Probabilities (Top 3):")
                    for location, probability in sorted(posterior.items(), key=lambda item: item[1], reverse=True)[:3]:
                        logger.info(f"    - {location}: {probability:.4f}")
                else:
                    logger.info("Could not calculate location belief.")

                # Expected Reward Calculation
                expected_reward = get_expected_reward(posterior, {"cpts_dir": str(cpts_dir)})
                logger.info(f"Expected Reward: {expected_reward:.4f}")

                # Distance from wall prediction
                p_distance_bins = b.get("p_distance_bins", {})
                if p_distance_bins:
                    expected_cm = sum((bin_idx + 0.5) * p for bin_idx, p in p_distance_bins.items())
                    logger.info(f"Predicted Distance from Wall: {expected_cm:.1f} cm")
                
                logger.info("-------------------\n")

                # --- Write to CSV ---
                pred_w.writerow([time.time(), predicted_location, "N/A", expected_cm, "N/A", expected_reward])
                pred_f.flush()

                distance_since_last_prediction = 0

                # REWARD-BASED TRIGGER for turning home
                if return_home and expected_reward > 0.8: # High confidence in being at the goal
                    logger.info(f"[predictor] Expected Reward > 0.8 ({expected_reward:.2f})! Turning around.")
                    
                    # Stop robot
                    await robot.set_wheel_speeds(STOP, STOP)
                    await asyncio.sleep(2)

                    ## Turn robot around
                    await turn_around_create3(robot, angle_deg=180, direction="left")

                    ## Swap which wall we follow
                    wall_side = "left" if wall_side == "right" else "right"
                    logger.info(f"[predictor] Turned 180°, now following {wall_side} wall")
                    returning = True
                    return_home = False

            await asyncio.sleep(dt)

    finally:
        await robot.set_wheel_speeds(0, 0)
        pred_f.close()
        logger.info(f"\n[predictor] Predictions saved to {predictions_csv_path.resolve()}")

async def turn_around_create3(robot: Create3, angle_deg: float = 180, direction: str = "left"):
    """
    Turn the robot in-place by angle_deg. Tries SDK turn_* first;
    falls back to a timed spin if those methods aren't available.
    """
    # Prefer SDK turn methods
    try:
        if hasattr(robot, "turn_left") and hasattr(robot, "turn_right"):
            if direction == "left":
                await robot.turn_left(angle_deg)
            else:
                await robot.turn_right(angle_deg)
            return
    except TypeError:
        # some SDK variants expose coroutine but with different signature
        pass

    # Fallback: timed spin using set_wheel_speeds. Calibrate yaw rate if needed.
    # Assumes wheel speed units consistent with your FORWARD/MAX_W/MIN_W.
    spin_w = max(0.5 * MAX_W, min(MAX_W, 0.6 * (MAX_W - MIN_W)))  # reasonable spin
    left = +spin_w if direction == "left" else -spin_w
    right = -spin_w if direction == "left" else +spin_w

    await robot.set_wheel_speeds(left, right)
    await asyncio.sleep(abs(angle_deg) / YAW_DEG_PER_SEC)
    await robot.set_wheel_speeds(0, 0)
