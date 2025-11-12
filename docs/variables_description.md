### Location (State Variable)
The primary state variable in our network. It represents the robot's belief about its current situation. The possible locations are:
*   **Wall:** The robot is following a continuous wall.
*   **Door_Start:** The robot is at the beginning edge of a doorway.
*   **Door:** The robot is currently inside a doorway.
*   **Door_Passed:** The robot has just passed the end edge of a doorway.

### Evidence Variables
These are the sensor readings that the network uses to infer the `Location`. Our model uses 45 evidence variables, representing a history of 9 time steps for 5 different sensor types.

*   **IR1-IR9:** The 9 most recent infrared sensor readings, discretized into bins. The discretization process is as follows:
    1.  The raw sensor reading is converted to a distance in centimeters using a logarithmic model: `distance = A + B * np.log(sensor_reading)`. The parameters `A` and `B` are determined by fitting the model to calibration data.
    2.  The distance is then discretized into 12 bins (0-11) of 1cm each. The distance in cm is converted to an integer, and the value is clamped between 0 and 11, inclusive. For example, a distance of 2.5cm would be in bin 2.
*   **PIDP1-PIDP9:** The 9 most recent Proportional error values from the PID controller.
*   **PIDI1-PIDI9:** The 9 most recent Integral error values from the PID controller.
*   **PIDD1-PIDD9:** The 9 most recent Derivative error values from the PID controller.
*   **BI1-BI9:** The 9 most recent bumper states (True/False).