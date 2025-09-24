class PID:
    def __init__(self, kp=0.4, ki=0.02, kd=0.1, setpoint=6.0):
        self.kp=kp; self.ki=ki; self.kd=kd
        self.setpoint=setpoint
        self.integral=0.0
        self.prev_error=None
    def reset(self):
        self.integral=0.0; self.prev_error=None
    def update(self, measurement, dt=0.1):
        """Return control signal given a distance measurement (cm)."""
        error = self.setpoint - measurement
        self.integral += error*dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error)/dt
        self.prev_error = error
        return self.kp*error + self.ki*self.integral + self.kd*derivative
