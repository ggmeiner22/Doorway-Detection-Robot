The core of this project is a Bayesian Network. This is a probabilistic model that allows us to reason about an unobserved state (the robot's location) based on observed sensor evidence.

Our network has a single hidden state variable, **Location**, which represents the robot's belief about its current situation (e.g., following a wall, entering a doorway).

The **Location** state is not directly measured. Instead, it is inferred from a series of 45 sensor readings (evidence variables) that are captured over the last 9 time steps. These evidence variables include:
*   **IR Sensor Readings:** Discretized distance from the wall.
*   **PID Controller Errors:** The proportional (P), integral (I), and derivative (D) error terms from the wall-following controller.
*   **Bumper State:** Whether a bumper has been pressed.

By using data from previous time steps as evidence, the Bayesian Network can reason about the robot's state based on patterns in the sensor readings, allowing it to make more robust decisions than it could with a single snapshot of data.