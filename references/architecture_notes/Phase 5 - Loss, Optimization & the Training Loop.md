## Assumptions & Logic:

* Data Formatting: We convert the NumPy arrays to PyTorch Tensors (torch.float32), which is the strict mathematical format required for hardware binding.

* Target Scaling (Gradient Collapse Prevention): We apply MinMaxScaler to the Y_train (RUL) targets, compressing massive cycle counts (e.g., 0 to 350) into a strictly positive 0.0−1.0 range. This stabilizes the initial loss calculations and prevents the optimizer from zeroing out the network weights due to exploding gradients.

* The Judge: We use Mean Squared Error (MSELoss). It is the industry standard for continuous regression. It measures the physical distance between predicted and actual RUL, squaring the error to aggressively penalize massive misses.

* The Optimizer: Adam (Adaptive Moment Estimation) dynamically adjusts the learning rate (weight adjustment aggression) for every individual CNN filter.

* Hyperparameters:
* * Batch Size (64): Dictates how many 3D sensor blocks the optimizer averages together before calculating error. 64 maximizes hardware caching efficiency.

* * Learning Rate (0.001): The mathematical sweet spot. Increasing causes violent divergence; decreasing causes glacial training times.

* * The baseline boundary between underfitting (failing to map the temperature curve to the lifespan) and overfitting (memorizing the exact statistical noise of the training data).