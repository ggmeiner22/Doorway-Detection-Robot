The Conditional Probability Tables (CPTs) are the "brain" of our Bayesian Network. Each table stores the probability of observing a certain sensor reading given the robot's current location. For example, `P(IR1=5 | Location=Door)`.

The CPTs are learned from the data you collect and are saved as individual CSV files in the `/cpts` directory. There is one prior probability file (`prior_location.csv`) and 45 CPT files, one for each evidence variable (e.g., `cpt_IR1.csv`, `cpt_PIDP1.csv`, etc.).

You can download the entire set of generated CPTs and the training data as zip archives.

*   [Download Standard CPTs (Zip)](docs/data_exports/cpts.zip) - For the original Naive Bayes model.
*   [Download POMDP CPTs (Zip)](docs/data_exports/cpts_pomdp.zip) - For the sequential POMDP model (Project 2).
*   [Download Data (Zip)](docs/data_exports/data.zip) - Raw collected sensor data.