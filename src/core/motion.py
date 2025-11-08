from .config import FORWARD

async def forward(robot):
    await robot.set_wheel_speeds(FORWARD, FORWARD)

async def backoff_left(robot):
    await robot.set_lights_on_rgb(255, 80, 0)
    await robot.move(-10)           # units: confirm cm vs mm
    await robot.turn_right(20)
    #await robot.forward

async def backoff_right(robot):
    await robot.set_lights_on_rgb(255, 80, 0)
    await robot.move(-10)
    await robot.turn_left(20)
    #await robot.forward
    
def front_obstacle_cm(sensors, approx):
    """Return min front distance in cm using your approximate_distance() callable."""
    return min(approx(sensors[3]), approx(sensors[4]))