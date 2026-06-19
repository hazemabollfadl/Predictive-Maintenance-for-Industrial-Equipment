## Assumptions & Logic:
* Extracting RUL: The RUL is the exact number of cycles remaining until failure. 
Since train_FD001.txt contains run-to-failure data, the formula is:
RUL = (Engine's Maximum Cycle) - (Current Cycle).

* The Regression Target: Deep Learning regression requires a continuous target instead of distinct state boundaries. We completely drop the W=30 warning window. The RUL integer itself is now the absolute target variable we want the model to predict.

* Validate: We will extract the final 5 cycles of Engine 1 to physically verify the RUL counts down exactly to 0.
