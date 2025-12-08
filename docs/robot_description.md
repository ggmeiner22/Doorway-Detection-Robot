The project is built on the iRobot Create 3, a programmable educational robot based on the Roomba i3 series platform.

### Motion Control
To navigate the hallway, the robot employs a **PID controller** that maintains a set distance of **6 cm** from the wall. This ensures the robot captures consistent sensor data for the perception system. The controller minimizes the error between the measured IR distance and the target setpoint, generating differential drive commands to correct deviations.

### Control Algorithm
The robot's behavior is driven by a hybrid architecture:
1.  **Low-Level Control (10 Hz):**
    *   Reads IR proximity.
    *   Calibrates raw reading to centimeters.
    *   Computes error `e(t) = d_set - d_cm`.
    *   Applies PID correction: `u(t) = K_p * e(t) + K_i * int(e) + K_d * de/dt`.
2.  **State Estimation (Spatial Trigger):**
    *   Updates belief distribution whenever the robot travels `approx. 10` cm.
    *   Checks the **Expected Reward** metric.
    *   If `E[R] > 0.8`, the robot stops, turns 180°, switches the wall-following side, and returns to the start.

### Key Sensors
*   **Infrared (IR) Sensors:** The Create 3 has a suite of 7 IR sensors. We use one of the front-facing side sensors as the primary input for our wall-following PID controller. The raw analog values are linearized into centimeters using a logarithmic calibration model: `d_cm = A + B * ln(raw)`.
*   **Bumper Sensors:** Two physical bumper zones allow detection of collisions. This is factored into the Bayesian Network as strong evidence that the robot is not in an open space.
*   **Wheel Encoders & IMU:** Used for odometry to track distance traveled and trigger spatial belief updates every 10cm.
