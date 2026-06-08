## Assumptions & Logic:
* Extracting RUL: The RUL is the exact number of cycles remaining until failure. 
Since train_FD001.txt contains run-to-failure data, the formula is:
RUL = (Engine's Maximum Cycle) - (Current Cycle).

* Create the binary State Mapping (W=30):
Deep Learning classification requires distinct state boundaries. We define the critical warning window (W) as 30 cycles.
If RUL > 30, the machine is Healthy (Class 0).
If RUL <= 30, the machine is in Imminent Failure (Class 1).

* Validate: We will extract the final 5 cycles of Engine 1 to physically verify the RUL counts down to 0 and the label holds steady at 1.
