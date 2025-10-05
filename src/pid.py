from dataclasses import dataclass

@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    setpoint: float = 0.0
    _integ: float = 0.0
    _prev: float | None = None

    def update(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self._integ += error * dt
        deriv = 0.0 if self._prev is None else (error - self._prev) / dt if dt > 0 else 0.0
        self._prev = error
        return self.kp*error + self.ki*self._integ + self.kd*deriv
