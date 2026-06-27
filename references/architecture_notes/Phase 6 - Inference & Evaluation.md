## Assumptions & Logic:

* The Extraction Shift (The Final Window): 
* * Training: We extracted every overlapping 30-cycle window to teach the complete degradation curve.
* * Testing: Because the test engines are suspended mid-flight, we strictly extract the final 30 cycles of each engine's timeline to predict its exact remaining life from that specific snapshot.

* The Leakage Trap (Standardization): WWe apply the exact StandardScaler fitted in Phase 3 using .transform(). Recalculating the mean/variance on the unseen test set is a catastrophic data leakage violation.

* The Answer Key Integration: We map the exact true integers from RUL_FD001.txt as our Y_test target array.

* Inverse Transformation: Because the network was trained on scaled targets (Phase 5), its raw output is a decimal (e.g., 0.35). We apply inverse_transform() to multiply this decimal back out to the real-world cycle scale before evaluation.

* The Evaluation Matrix (RMSE & MAE): We bypass classification metrics entirely. 
Mean Absolute Error (MAE): Defines the average physical cycle deviation (how many cycles the prediction was off by).
Root Mean Square Error (RMSE): Aggressively penalizes severe prediction outliers. Used as the primary benchmark for model leverage and safety in industrial contexts.