### Dependencies
The project requires Python 3 and the `irobot_edu_sdk`. You can install the necessary libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Source Code Files
*   **Data Collection:** <a href="src/scripts/collect.py" download>collect.py</a> - Run this script to drive the robot and manually collect labeled training data.
*   **CPT Training:** <a href="src/scripts/train.py" download>train.py</a> - Run this script to learn the Conditional Probability Tables (CPTs) from the collected CSV data.
*   **Main Execution:** <a href="src/scripts/run.py" download>run.py</a> - Run this script to have the robot follow a wall and make predictions using the trained network.
*   **Belief Network Core:** <a href="src/core/belief_network.py" download>belief_network.py</a> - Contains the core `belief` function that implements the Bayesian sensor fusion.
*   **Robot Control Logic:** <a href="src/core/control.py" download>control.py</a> - Contains the main functions for controlling the robot's behavior.