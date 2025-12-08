### Perception and Inference
The robot constructs a belief about its location using a **Partially Observable Markov Decision Process (POMDP)** framework. Continuous sensor data—including IR readings and the PID controller's own internal state—is discretized and fed into a Bayesian filter. This filter updates the probability of the robot being in specific states (e.g., "Wall", "Door", "Door Passed") based on a learned observation model and a transition model that enforces forward progression.

### POMDP Formulation
We formalize the navigation task as a POMDP defined by the tuple (S, A, T, Omega, O, R). The robot cannot directly observe its true state `s_t` in `S` but maintains a belief distribution `b_t(s) = P(s_t = s | z_{1:t}, a_{1:t})`.

#### Topological State Space and Parameter Tying
The hallway is modeled as a directed acyclic graph (DAG) representing the linear sequence of "Macro-States" (Wall, Door 1, Wall, Door 2, etc.).
The global state space `S` consists of specific localized states:
`S = { Wall_0, Start_1, Door_1, Passed_1, Wall_1, ..., Wall_End }`

To avoid overfitting, we employ **Parameter Tying**. All states of a similar semantic class share the same Conditional Probability Tables (CPTs). We define a mapping `M: S -> S_generic` where `S_generic = { Wall, DoorStart, Door, DoorPassed, WallEnd }`.

#### Transition Dynamics
The transition model `T(s' | s)` encodes the topology of the environment:
*   **Persistent States (Wall, Door):** Modeled with a self-transition probability (`P_stay = 0.8`) and a progression probability (`P_next = 0.2`). This models the variable length of walls and door openings.
*   **Transient States (DoorStart, DoorPassed):** Modeled with deterministic transitions (`P_next = 1.0`) to the subsequent state.

This structure enforces a strict sequential progression: the robot cannot skip a door or move backward, constraining the inference search space significantly.

### Observation Model
The observation probability `P(z_t | s)` is computed using a **Naive Bayes** assumption to handle the high dimensionality of the feature vector `z_t`.
The underlying CPTs `P(f_i | c)` for each generic class are learned from labeled training data via Maximum Likelihood Estimation with Laplace smoothing (`alpha=1`).

The belief update is performed recursively via the discrete Bayes filter:
`b_t(s') = eta * P(z_t | M(s')) * sum_{s in S} T(s' | s) b_{t-1}(s)`
where `eta` is the normalization constant.

### Decision Policy (Expected Reward)
The ultimate goal is to execute a discrete high-level action: turning around after passing 3 doors.
Rather than relying on the most likely state (MAP estimate), we utilize an **Expected Reward** policy.
We augment the feature space with a latent "Reward" variable. During training, the `Wall_End` state is associated with a high probability of `Reward=1`, while all other states have `Reward=0`.
At runtime, we compute the expected reward `E[R_t]`:
`E[R_t] = sum_{s in S} b_t(s) * P(Reward=1 | M(s))`
When this metric exceeds a threshold (`tau = 0.8`), the robot executes the return-home maneuver.
