# src/drive_test.py
import asyncio, os
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3, event

NAME = os.environ.get("IROBOT_NAME", "iRobot-88FA7A7E3FCC461E8B675C")
robot = Create3(Bluetooth())

@event(robot.when_play)
async def play(robot):
    print("Driving forward...")
    await robot.set_lights_on_rgb(0,255,0)
    await robot.set_wheel_speeds(10, 10)   # 100 mm/s
    await asyncio.sleep(3)
    await robot.set_wheel_speeds(0, 0)
    await robot.set_lights_on_rgb(255,0,0)
    print("Done.")

robot.play()
