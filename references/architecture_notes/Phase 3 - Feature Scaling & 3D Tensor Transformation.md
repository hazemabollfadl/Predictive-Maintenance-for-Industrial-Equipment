## Assumptions & Logic:
* For this one we're gonna start by applying Standard Scaler to the 15 Active sensors we have.
**Why?**
Feeding the model a very large value of one sensor "Sensor 4 (Temperature): 1400" and another very small value "Sensor 7 (Pressure): 39", Would lead the model to assume that sensor 4 is the most important feature which isn't true by nature.
The network will spend all its compute cycles obsessing over minor changes in Sensor 4, while completely ignoring Sensor 15.
**The fix**
Z-Score Standardization, It destroys the original units (degrees, PSI, RPM) and replaces them with a universal metric: Standard Deviations from the Mean.
```math
Z = X−μ/σ
X: The raw sensor reading.
μ: The average of that sensor across the training data.
σ: The standard deviation of that sensor.
```
**Two step process**
* * First, We calculate the average of the sensor, and we subtract that exact average from every single row in that column. (X - Mean)
If the temperature is 1400 today, and the average is 1400, the new value is 0. If the temperature drops to 1390, the new value is -10. If it spikes to 1420, the new value is +20.
The baseline '0': Becomes normal, Positive: Becomes an increase, Negative: naturally a decrease

* * second, Sensor 4 might change by 20 degrees normally. Sensor 15 might change by 0.1 normally.
If Sensor 15 jumps by 0.5, that is a arounf 500% spike, but the Model Network will ignore a 0.5 if it is constantly looking at +20 change from Sensor 4.
Now we divide the centered numbers we got from above by that standard deviation. (X - Mean) / StdDev
Result: if Sensor 4's standard deviation is 20, and it spikes by +20 degrees, dividing it by 20 equals 1.0. If Sensor 15's standard deviation is 0.1, and it spikes by +0.1, dividing it by 0.1 equals 1.0.


two example to visually understand:
```math
Sensor A: reading is 1420. (Mean is 1400, StdDev is 20).
Standardized Math: (1420−1400)/20=1.0

Sensor B: Raw reading is 42. (Mean is 40, StdDev is 2).
Standardized Math: (42−40)/2=1.0
```


* 3D transformation, Argubally the trickiest part of this process we're gonna divide this process a bit.
* * First We'll define W=30 cycles. The CNN will look at 30 consecutive cycles to make a single prediction.
* * Secondly, We will drop the first 29 cycles of every engine. We can't look back 30 cycles when we are at Cycle 15.
* * We extraced 17,731 indvidual cycles, each of them is 30 row long denoting the 30 cycle limit we applied with 15 Sensors reading in each.
* * We extracted the target label of every cycle iteration of every Engine and saved them.

