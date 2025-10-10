from __future__ import annotations
import asyncio
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from src.core.control import collect_data_manual
from src.core.config import DATA_CSV, ROBOT_NAME


async def main():
    robot = Create3(Bluetooth(ROBOT_NAME))
    await collect_data_manual(robot, data_csv=DATA_CSV)

if __name__ == "__main__":
    asyncio.run(main())