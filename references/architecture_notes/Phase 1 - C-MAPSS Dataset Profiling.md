## Assumptions & Logic:
* C-MAPSS contains four sub-datasets (FD001 to FD004) with varying operating conditions and fault modes. We strictly use train_FD001.txt. It contains 1 operating condition (Sea Level, Mach 0, constant Throttle Resolver Angle of 100) and 1 fault mode (High-Performance Compressor Degradation). In practical terms, the simulated engine is bolted to a ground test stand, running at a steady maximum thrust until it destroys itself. This minimizes mathematical complexity while building the core 1D CNN logic.

* The data features no standard timestamps. Time is measured continuously in operational Cycles.

* End-of-Life (EOL) Assumption: The dataset consists of run-to-failure simulations. We assume the last recorded cycle for any given Engine ID represents the exact point of catastrophic failure.
    
* Sensor Variance Logic: Not all 21 sensors carry a degradation signal. Sensors with a variance of zero (flatlines) represent constant operational settings. They must be identified and purged immediately to reduce dimensionality and compute friction.

## Execution Steps
* Acquisition: Load the raw .txt file into a structured Pandas DataFrame. Assign the correct column mapping (Engine_ID, Cycle, OpSet1-3, Sensor1-21).

* Group Analytics: Calculate the maximum lifespan (EOL) for each Engine ID.

* * In the training set: NASA provides run-to-failure data but no explicit Remaining Useful Life (RUL) file. We engineer the RUL by finding the Maximum Cycle (EOL) for each engine and counting backward. (e.g., If Engine 1 fails at Cycle 192, its RUL at Cycle 100 is 92).

* * In the testing set: The engines are "suspended" mid-flight. To mathematically verify the model, we use the external RUL_FD001.txt answer key, which contains the true remaining cycles for these suspended engines.

* Variance Profiling: Calculate the standard deviation for all 21 sensors. Isolate and drop sensors that carry zero or near-zero variance.

* Visual Verification: Plot a high-variance sensor to visually confirm the exponential degradation signal.