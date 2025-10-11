The project is built on the iRobot Create 3, a programmable educational robot based on the Roomba i3 series platform. It is designed for students and developers to explore robotics, computer science, and engineering.

For this project, we primarily leverage its onboard sensor suite to achieve autonomous wall-following and location detection. The key sensors and features we use are:

*   **Infrared (IR) Sensors:** The Create 3 has a suite of 7 IR sensors that it uses for obstacle avoidance. We use one of the front-facing side sensors as the primary input for our wall-following PID controller, allowing us to precisely measure the robot's distance from the wall.

*   **Bumper Sensors:** Two physical bumper zones on the front of the robot allow it to detect direct contact with obstacles. A bumper press is a critical event that is factored into our Bayesian Network as strong evidence that the robot is not in an open space.

*   **Wheel Encoders & IMU:** The robot's two wheel encoders, combined with its Inertial Measurement Unit (IMU) which includes a gyroscope and accelerometer, provide odometry data. This allows us to track the robot's position and calculate the distance it has traveled.

The robot also features a programmable RGB LED light ring, which we use to provide visual feedback about the robot's current state and predictions.