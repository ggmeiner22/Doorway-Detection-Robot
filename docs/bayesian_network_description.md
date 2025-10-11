The core of this project is a Bayesian Network, a powerful probabilistic model for reasoning under uncertainty. In robotics, Bayesian Networks are ideal for **sensor fusion**—the process of combining data from multiple, often noisy, sensors to arrive at a more accurate and reliable understanding of the world.

Our network is designed to infer the robot's unobserved **Location** (e.g., `Wall`, `Door`) by observing evidence from its sensors over time. Here's how it works:

*   **Nodes are Variables:** The "parent" node in our network represents the `Location`, which is the hidden state we want to find. "Child" nodes represent our 45 evidence variables (the historical sensor readings from the IR, PID controller, and bumpers).

*   **Edges are Dependencies:** The structure of the network defines the dependencies. We use a Naive Bayes structure, which assumes that each piece of sensor evidence is conditionally independent of the others, given the robot's location. This is a common and effective simplification for this type of problem.

*   **Probabilities are Key:** Each node has a Conditional Probability Table (CPT) that stores the probability of observing a certain value given its dependencies (if there are any). These CPTs are learned from the data we collected.

As the robot moves, it gathers new sensor readings. The Bayesian Network then performs **inference**, updating its belief in the probability of each possible `Location` by combining the prior belief with the new evidence using the CPTs. This allows the robot to weigh all the evidence and make an informed decision about its state, even with imperfect sensor data.