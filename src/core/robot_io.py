from __future__ import annotations
import os, asyncio
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3

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
            return 50.0
        return min(round(distance, 2), 50.0)
    else:
        return 50.0

def cm_to_bin12(cm: float) -> int:
    """Discretizes a distance in cm into 12 bins (0-11) of 1cm each."""
    return int(max(0, min(11, cm)))

def discretize_p_10_bins(measurement: float, setpoint: float) -> int:
    if setpoint <= 0:
        return 5
    val = 2 - (measurement / setpoint)
    clamped_val = max(0.0, min(val, 2.0))
    norm_val = clamped_val / 2.0
    binned_val = int(norm_val * 10)
    return max(0, min(binned_val, 9))

def discretize_i_10_bins(integral_val: float, setpoint: float) -> int:
    if setpoint <= 0:
        return 5
    T = 0.5 * setpoint
    if T <= 0:
        return 5
    clamped_val = max(-T, min(integral_val, T))
    norm_val = (clamped_val / (2 * T)) + 0.5
    binned_val = int(norm_val * 10)
    return max(0, min(binned_val, 9))

def discretize_d_10_bins(derivative_val: float, setpoint: float) -> int:
    if setpoint <= 0:
        return 5
    T = 0.1 * setpoint
    if T <= 0:
        return 5
    clamped_val = max(-T, min(derivative_val, T))
    norm_val = (clamped_val / (2 * T)) + 0.5
    binned_val = int(norm_val * 10)
    return max(0, min(binned_val, 9))
