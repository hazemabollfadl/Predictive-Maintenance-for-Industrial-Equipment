## Assumptions & Logic:
*We apply StandardScaler exclusively to the active sensors.
Why?
Feeding the model a massive value from one sensor (e.g., Temperature: 1400) and a tiny value from another (e.g., Pressure: 39) forces the network to assume the larger number is more important. The network will expend its compute cycles obsessing over minor temperature changes while ignoring critical pressure drops.
The Fix: Z-Score Standardization
It destroys the original physical units (Degrees, PSI, RPM) and replaces them with a universal metric: Standard Deviations from the Mean.
```math
Z = X−μ/σ
X: The raw sensor reading.
μ: The average of that sensor across the training data.
σ: The standard deviation of that sensor.
```
**Two step process**
* * Subtract the average of the sensor from every row in that column (X−μ). The baseline becomes 0 (normal). Positive becomes an increase; negative becomes a decrease.

* * Divide the centered numbers by that sensor's standard deviation. This normalizes the scale of volatility across all sensors.


two example to visually understand:
```math
Sensor A: reading is 1420. (Mean is 1400, StdDev is 20).
Standardized Math: (1420−1400)/20=1.0

Sensor B: Raw reading is 42. (Mean is 40, StdDev is 2).
Standardized Math: (42−40)/2=1.0
```


* 3D transformation:
* * Define W=30 cycles. The CNN evaluates 30 consecutive cycles to make a single RUL prediction.
* * Slide a window down the timeline. We drop the first 29 cycles of every engine because a full 30-cycle retrospective is mathematically impossible before Cycle 30.
* * We extract individual 3D blocks (shape: 30 rows, 14 sensors) and align them with the continuous RUL target located at the final row of that specific block.
