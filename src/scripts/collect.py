import time, threading
from irobot_edu_sdk.robots import event
from irobot_edu_sdk.music import Note
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from src.core.config import CPTS_DIR, DT, ROBOT_NAME, ensure_dirs, DOOR_STATES, DATA_CSV
from src.core.control import run_controller,collect_data_manual

robot = Create3(Bluetooth(ROBOT_NAME))
print("[run] Using event-driven SDK loop")

@event(robot.when_play)
async def play(robot):
    await robot.set_lights_on_rgb(30, 255, 100)
    await robot.play_note(Note.A5, .5)
    print(f"[run] Connected to {ROBOT_NAME}")
    ensure_dirs()

    #Collect Data
    await collect_data_manual(robot, data_csv=DATA_CSV)

robot.play()
