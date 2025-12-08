### Feature Engineering and Discretization
To robustly identify environmental features despite the low fidelity of individual sensors, we construct a high-dimensional observation vector that captures both the external state of the world and the internal "effort" of the robot's controller.
At each discrete time step $t$ (corresponding to approximately **10 cm** of forward travel), the system aggregates sensor readings into a feature vector $z_t$.
To capture temporal context—essential for distinguishing instantaneous sensor noise from persistent geometric features like doors—we maintain a sliding history window of length $N=9$. The feature vector for a single time step is thus a concatenation of 54 distinct values:
`z_t = { IR, PID_P, PID_I, PID_D, ODO, Bumper }` for the last 9 steps.

### Signal Preprocessing
To facilitate efficient inference in the discrete Bayesian network, all continuous signals are mapped to finite integer bins. This mapping is calibrated to maximize information gain in the critical regions around the setpoint.

#### 1. IR Proximity (IR)
*   **Source:** Side-facing IR sensor.
*   **Processing:** Raw analog values are linearized into centimeters using a logarithmic calibration model derived from empirical data: $d_{cm} = A + B \ln(raw)$.
*   **Discretization:** Clamped to the range `[0, 11]` cm and discretized into **12 unit bins**.

#### 2. PID Controller States (Proprioception)
The internal error terms of the PID wall-follower serve as "proprioceptive" sensors. By looking for patterns in integral windup and fast changes to the derivative, we aim to identify unique features of the hallway.
*   **Proportional (P):** The error $e(t)$ is normalized relative to the setpoint. Mapped to **10 bins**. The mapping centers the ideal state (error=0) in the middle of the bin range (Bin 5).
*   **Integral (I):** The accumulated error is clamped to a range of $\pm 0.5 d_{set}$ before being mapped to **10 bins**. This feature is particularly sensitive to the "open" space of a doorway.
*   **Derivative (D):** The rate of change is clamped to $\pm 0.1 d_{set}$ and mapped to **10 bins**. This captures the high-frequency edges of the doorframe.

#### 3. Odometry (ODO)
*   **Source:** Wheel encoders.
*   **Processing:** Verifies the Euclidean distance traveled during the update step to account for variations in velocity (e.g., due to friction).
*   **Discretization:** Mapped to **6 bins** centered around the target step size of 10 cm.

#### 4. Bumper (BI)
*   **Source:** Front bumper contact.
*   **Discretization:** Binary (True/False).
