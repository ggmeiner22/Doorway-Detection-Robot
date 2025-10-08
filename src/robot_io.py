from __future__ import annotations
import os, asyncio
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit

try:
    from irobot_edu_sdk.backend.bluetooth import Bluetooth
    from irobot_edu_sdk.robots import Create3
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: irobot_edu_sdk.\n"
        "Windows:  python -m pip install irobot_edu_sdk bleak winsdk numpy scipy\n"
        "Linux:    python -m pip install irobot_edu_sdk bleak numpy scipy\n"
        "WSL is not supported for BLE."
    ) from e

def get_robot(target_name_or_mac: Optional[str]) -> Create3:
    """Return a connected Create3 using Bluetooth backend."""
    return Create3(Bluetooth(target_name_or_mac))

# Model function for converting sensor readings to distance
def model_function(reading, A, B):
    return A + B * np.log(reading)

# Calibration data for IR sensors
readings = np.array([2525, 1560, 1060, 660, 520, 350, 270, 200, 150, 120, 90, 70, 50, 40, 30, 24, 15])
distances = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9])
params, _ = curve_fit(model_function, readings, distances)
A, B = params

def prox_to_cm(sensor_reading: float) -> float:
    if sensor_reading > 0:
        distance = A + B * np.log(sensor_reading)
        if np.isinf(distance) or np.isnan(distance):
            return 999.0
        return round(distance, 2)
    else:
        return 999.0

def cm_to_bin10(cm: float) -> int:
    return int(max(0, min(9, cm // 10)))