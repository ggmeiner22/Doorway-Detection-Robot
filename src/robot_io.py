from __future__ import annotations
import os, asyncio
from typing import Optional

try:
    from irobot_edu_sdk.backend.bluetooth import Bluetooth
    from irobot_edu_sdk.robots import Create3
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: irobot_edu_sdk.\n"
        "Windows:  python -m pip install irobot_edu_sdk bleak winsdk numpy\n"
        "Linux:    python -m pip install irobot_edu_sdk bleak numpy\n"
        "WSL is not supported for BLE."
    ) from e

def get_robot(target_name_or_mac: Optional[str]) -> Create3:
    """Return a connected Create3 using Bluetooth backend."""
    return Create3(Bluetooth(target_name_or_mac))

async def set_led_for_label(robot: Create3, label: str):
    if label == "Wall":          await robot.set_lights_on_rgb(0,255,0)
    elif label == "Door":        await robot.set_lights_on_rgb(0,0,255)
    elif label == "Door_Start":  await robot.set_lights_on_rgb(255,255,0)
    elif label == "Door_Passed": await robot.set_lights_on_rgb(255,0,255)

def prox_to_cm(raw: float) -> float:
    try:
        r = float(raw)
    except Exception:
        return 999.0
    return max(5.0, min(150.0, 2000.0 / max(1.0, r)))

def cm_to_bin10(cm: float) -> int:
    return int(max(0, min(9, cm // 10)))
