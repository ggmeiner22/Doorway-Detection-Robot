"""
This script implements the door counting and navigation task described in
"Counting Doors by Integrating POMDP Models and PID Controllers".
It uses a location predictor based on a POMDP, which includes a transition model,
to pass three doors and then turn around.

Run with:
    python -m src.scripts.runPOMDP
"""
from irobot_edu_sdk.robots import event
from irobot_edu_sdk.music import Note
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from src.core.config import CPTS_DIR, DT, ROBOT_NAME, ensure_dirs, DOOR_STATES
from src.core.control import run_location_predictor_pomdp
from src.core.pomdp_belief_network import pomdp_belief

robot = Create3(Bluetooth(ROBOT_NAME))
print("[runPOMDP] Using event-driven SDK loop")

@event(robot.when_play)
async def play(robot):
    await robot.set_lights_on_rgb(30, 255, 100)
    await robot.play_note(Note.A5, .5)
    print(f"[runPOMDP] Connected to {ROBOT_NAME}")
    ensure_dirs()

    # This function uses the POMDP belief network to predict the robot's location,
    # count doors, and turn around after passing the third door.
    await run_location_predictor_pomdp(robot, cpts_dir=CPTS_DIR, door_states=DOOR_STATES, belief_func=pomdp_belief, dt=DT, return_home=True)


robot.play()