## Assumptions & Logic:
* Extracting RUL: The RUL is the exact number of cycles remaining until failure. Since train_FD001.txt contains run-to-failure data, the formula is: RUL = (Engine's Maximum Cycle) - (Current Cycle).

* The Regression Target: Deep Learning regression requires a continuous target. The warning window (W=30) classification logic is dropped. The RUL integer itself serves as the absolute target variable the network must predict.

* Validate: Extract the final 5 cycles of an engine to physically verify the RUL counts down exactly to 0.