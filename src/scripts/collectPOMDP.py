import time
from irobot_edu_sdk.robots import event
from irobot_edu_sdk.music import Note
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from src.core.config import ROBOT_NAME, ensure_dirs, DATA_CSV
from src.core.control import collect_data_pomdp
from src.core.motion import backoff_right, backoff_left

robot = Create3(Bluetooth(ROBOT_NAME))
print("[run] Using event-driven SDK loop")

# Register event handlers ***after*** robot is created
@event(robot.when_bumped, [False, True])   # right bumper
async def avoidcollision_right(robot_event):
    print("Right bumper: backoff")
    await backoff_right(robot)
    await robot.set_lights_on_rgb(30, 255, 100)

@event(robot.when_bumped, [True, True])    # left bumper
async def avoidcollision_left(robot_event):
    print("Left bumper: backoff")
    await backoff_left(robot)
    await robot.set_lights_on_rgb(30, 255, 100)

@event(robot.when_play)
async def play(robot):
    await robot.set_lights_on_rgb(30, 255, 100)
    await robot.play_note(Note.A5, .5)
    print(f"[run] Connected to {ROBOT_NAME}")
    ensure_dirs()

    # Generate a unique filename for the data collection session
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_csv = DATA_CSV.parent / f"measurements_pomdp_{timestamp}.csv"

    #Collect Data
    await collect_data_pomdp(robot, data_csv=session_csv)

robot.play()
