from __future__ import annotations
import asyncio
from irobot_edu_sdk.robots import Create3
from .pid import PID
from .robot_io import prox_to_cm, cm_to_bin10
from .belief_network import belief, door_passed_10cm_ago
from .config import DT, SETPOINT_CM, FORWARD, MAX_W, MIN_W, RIGHT_IR_IDX

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
