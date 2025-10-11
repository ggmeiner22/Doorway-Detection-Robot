The project uses the iRobot Create 3 educational robot. This robot is equipped with a variety of sensors, but our project primarily utilizes:

*   **Infrared (IR) Sensors:** The robot has 7 IR sensors. We use one of the front-facing side sensors to measure the distance to the wall for our wall-following PID controller.
*   **Bumper Sensors:** Two physical bumper sensors are located on the front of the robot. These are used to detect collisions. A bumper press is a key event that is factored into our Bayesian Network.
*   **Wheel Encoders:** These sensors track the rotation of the wheels, providing odometry data that allows us to calculate the distance the robot has traveled. This is used to trigger predictions.