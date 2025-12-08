### Project Overview
The "Doorway-Detection-Robot" project, now in its **Milestone 3** phase, builds upon previous work (Milestones 1 & 2) to implement a sophisticated autonomous navigation system. Utilizing a **Partially Observable Markov Decision Process (POMDP)** framework, the robot executes a complex, sequential mission: identify and count three specific doors on its right side, stop at the third location to perform a task (e.g., cargo acquisition), and then autonomously return to its starting point. This system is designed to robustly manage the uncertainties inherent in sensor data and partial environmental observability.

### Framework
The system is organized as a modular pipeline that connects the iRobot Create3 hardware, a probabilistic inference model, and a closed-loop controller. The architecture is composed of four major layers:
1.  **Data Collection**
2.  **Model Learning**
3.  **Online State Estimation**
4.  **Control and Task Logic**

Each layer communicates through well-defined interfaces, making it possible to update one component without redesigning the entire system.

*   **Data Collection Layer:** Responsible for sending motion commands to the robot over Bluetooth and collecting all available sensory input (infrared distance readings, bumper states, and wheel odometry). These are recorded into CSV logs with human-provided location labels and reward annotations.
*   **Model Learning Layer:** Processes the recorded datasets by discretizing raw sensor values and estimating observation likelihood tables (CPTs) for the POMDP features.
*   **Online State Estimation Layer:** Maintains a belief of the robot’s location using a Bayesian update mechanism. It uses short histories of discretized features and learned CPTs to infer the most probable location state.
*   **Control Layer:** Handles movement and task completion. The PID controller adjusts movement based on IR distance, while the belief estimate triggers high-level actions (like turning around after passing three doors).

### Experiment
The goal of the experimental evaluation was to determine whether the hybrid POMDP-PID architecture enables reliable door-passage detection while maintaining stable wall-following behavior.

The experiment consisted of three phases:
1.  **Data Collection:** The robot was manually operated along the hallway to gather labeled training data.
2.  **Model Learning:** Logs were discretized into 54-dimensional feature vectors, and Parameter Tying was applied to learn "Macro-States" (Wall, DoorStart, Door, DoorPassed).
3.  **Online Evaluation:** The robot performed four autonomous trials. Completion success was defined as detecting all three doors, surpassing the expected-reward threshold of `0.8`, and executing the return maneuver.

### Results
The experimental evaluation demonstrates that the POMDP-PID hybrid architecture effectively enables autonomous door counting and navigation. The system achieved a **91.7% door detection rate** across four trials, successfully identifying all three doorways in three of four tests. Successful navigation completion was achieved in **75%** of trials.

#### 1. Door Detection Performance
The POMDP-based localization system demonstrated robust door detection capabilities.
*   **Total Doors Encountered:** 12 (3 doors x 4 trials)
*   **Successfully Detected:** 11
*   **Detection Rate:** 91.7%
*   **False Positives:** 0.75 (avg per trial)
*   **False Negatives:** 0.25 (avg per trial)

#### 2. Wall-Following Stability and Feature Quality
The PID controller maintained a highly stable lateral offset, reflected in the conditional probability tables (CPTs).
The PID error values are discretized into 10 bins (0-9), where the central bin (5) corresponds to zero error (the setpoint). Bins 0-4 represent negative error (robot is farther than the setpoint), while bins 6-9 represent positive error (robot is closer than the setpoint).
This is clearly seen in the proportional feature `PIDP9`. In `Wall` states, the probability mass is heavily concentrated in the central bins (5 and 6), summing to approximately **0.84** (0.44 and 0.40 respectively). In contrast, during the `Door` state, the distribution shifts significantly, with bins 7, 8, and 9 accounting for **~0.65** of the probability. This distinct shift captures the controller's immediate reaction to the discontinuity in the wall, serving as a reliable proprioceptive cue for the POMDP.

#### 3. Online Door Detection & Expected Reward
The online prediction logs indicate that the POMDP belief tracker reliably identified door regions. The belief transitions were smooth and monotonic.
The `Wall_End` state was uniquely characterized by the reward signal. Analysis of `cpt_Reward.csv` shows the probability of receiving a high reward given `Wall_End` is > 0.99, versus near zero for other states.
In online logs, the **Expected Reward** metric tracked this precisely. As the robot passed the third door, the expected reward jumped from negligible values (~approx. ~0.006) to **0.997** upon entering `Wall_End`. This signal triggered the return-home maneuver.
