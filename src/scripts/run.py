#!/usr/bin/env python3
"""Orchestrator: collect -> train CPTs -> control (Create3 over BLE).

Run with:
    python -m src.run.py
"""
import time, threading
from irobot_edu_sdk.robots import event
from irobot_edu_sdk.music import Note
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from src.core.config import CPTS_DIR, DT, ROBOT_NAME, ensure_dirs, DOOR_STATES
from src.core.control import run_location_predictor

robot = Create3(Bluetooth())
print("[run] Using event-driven SDK loop")

@event(robot.when_play)
async def play(robot):
    await robot.set_lights_on_rgb(30, 255, 100)
    await robot.play_note(Note.A5, .5)
    print(f"[run] Connected to {ROBOT_NAME}")
    ensure_dirs()

    # Phase 3: control with BN belief for location prediction
    await run_location_predictor(robot, cpts_dir=CPTS_DIR, door_states=DOOR_STATES, dt=DT)


robot.play()