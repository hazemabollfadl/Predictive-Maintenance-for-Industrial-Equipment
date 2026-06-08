## Assumptions & Logic:
* C-MAPSS contains four sub-datasets (FD001 to FD004) with varying operating conditions and fault modes. We will strictly use 
train_FD001.txt. It contains 1 operating condition (represents an engine running at Sea Level, with a Mach number of 0, and a constant Throttle Resolver Angle (TRA) of 100. In practical terms, it means the simulated engine is bolted to a test stand on the ground, running at a steady, maximum thrust until it destroys itself) and 1 fault mode (HPC - High permformance Compressor Degradation). 
This minimizes from the mathematical complexity form the other three datasets while we build the core 1D CNN logic.

* The data has no standard timestamps. Time is measured in Cycles.

* End-of-Life (EOL) Assumption: The dataset consists of run-to-failure simulations. We assume the last recorded cycle for any given Engine ID is the exact point of catastrophic failure.

* Sensor Variance Logic: We assume not all 21 sensors carry signal. 
Sensors with a variance of zero (flatlines) represent constant operational settings, not degradation. They must be identified and purged immediately to reduce dimensionality.

## Execution Steps
* Acquisition: Load the raw .txt file into a structured Pandas DataFrame. Assign correct column names (Engine_ID, Cycle, OpSet1-3, Sensor1-21).

* Group Analytics: Calculate the maximum lifespan (EOL) for each Engine ID to understand the distribution of engine failure times.
Why do we need this intead of using the RUL?

* * In the training set: We know the engine died on its very last recorded row, NASA did not provide an RUL file for it. You are expected to engineer the RUL yourself by finding the Maximum Cycle (EOL) for each engine and counting backward. If Engine 1's last row is Cycle 192, its EOL is 192. At Cycle 100, its RUL was 92.

* * In the testing set: Every single engine in this file is "Suspended."
The Logic: NASA paused the simulation mid-flight. The engine has not failed yet. Engine 1 might stop at Cycle 31. Engine 2 might stop at Cycle 140.
How do you mathematically prove your model is right if the test file doesn't show the failure?

* * The Answer is RUL_FD001.txt: -
This file contains a single column of numbers. These are the true remaining cycles for the suspended engines in the test set. If Row 1 in this file says 112, it means Engine 1 in the test set actually had exactly 112 cycles left before it would have exploded.

* Variance Profiling: Calculate the standard deviation for all 21 sensors. Identify and isolate sensors that carry zero or near-zero variance and remove them.

* Visual Verification: Plot a high-variance sensor to visually confirm the degradation signal.