### Location (State Variable)
The primary state variable in our network. It represents the robot's belief about its current situation. The possible locations are:
*   **Wall:** The robot is following a continuous wall.
*   **Door_Start:** The robot is at the beginning edge of a doorway.
*   **Door:** The robot is currently inside a doorway.
*   **Door_Passed:** The robot has just passed the end edge of a doorway.

### Evidence Variables
These are the sensor readings that the network uses to infer the `Location`. Our model uses 45 evidence variables, representing a history of 9 time steps for 5 different sensor types.

*   **IR1-IR9:** The 9 most recent infrared sensor readings, discretized into bins.
*   **PIDP1-PIDP9:** The 9 most recent Proportional error values from the PID controller.
*   **PIDI1-PIDI9:** The 9 most recent Integral error values from the PID controller.
*   **PIDD1-PIDD9:** The 9 most recent Derivative error values from the PID controller.
*   **BI1-BI9:** The 9 most recent bumper states (True/False).