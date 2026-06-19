## Assumptions & Logic:

* The Extraction Shift (The Final Window): 
* * Training: We extracted every overlapping 30-cycle window to teach the model the entire continuous degradation curve.
* * Testing: The test engines are suspended mid-flight. We strictly extract the final 30 cycles of each engine's recorded timeline to predict its exact remaining life from that specific snapshot in time.

* The Leakage Trap (Standardization): We apply the exact scaler object fitted in Phase 3 using .transform(). Recalculating the mean/variance on the test set is a catastrophic data leakage violation.

* The Answer Key Integration: We load the RUL_FD001.txt file. These are the true integer remaining cycles for the 100 suspended test engines. We map these directly as our Y_test target array.

* The Evaluation Matrix (RMSE & MAE): We bypass classification metrics entirely. 
We deploy Mean Absolute Error (MAE) to understand the average physical cycle deviation, and Root Mean Square Error (RMSE) to aggressively penalize severe prediction outliers.