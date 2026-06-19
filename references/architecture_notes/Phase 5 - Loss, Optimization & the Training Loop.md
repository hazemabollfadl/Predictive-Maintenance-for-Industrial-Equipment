## Assumptions & Logic:

* Convert the numpy arrays we got from Phase_3 to Tensorflow float32 since that what PyTorch requires it

* We use Mean Squared Error (MSELoss) as our Loss Function. It is the industry standard for continuous regression. The goal is to measure the physical distance between the predicted Remaining Useful Life (RUL) and the actual RUL, square that error to penalize massive misses, and minimize it to increase our model's precision.


* As an optimizer for the learning rate we use Adam (Adaptive Moment Estimation). it dynamically adjusts the "learning rate" (how aggressively it changes the weights) for every single CNN filter.

* Create the learning loop (Backpropagation):
These are the core operations of DL it's a 5 step process per batch:
forward(): Make a prediction.
loss(): Calculate the error.
zero_grad(): Clear out the old adjustments from the previous batch.
backward(): Calculate the exact mathematical adjustments needed to fix the error (Derivatives).
step(): Actually apply those adjustments to the network's weights.

* Some numbers:
* * We will do batches of 64 to be kind to our machines, it's how many cycles Adam optimizer calculates the error and updates the weights
* * the learning rate of 0.001 is standard here, increasing or decreasing this will tell Adam to either aggresively overcorrect the loss function or take alot of time to learn and update the weights
* * 15 Epochs is also a sweetspot, increasing or decreasing this results in the model 
overfitting(It has seen the data way too much and starting to memorize instead of predict)
underfitting(It hasn't seen the data enough times to figure out how a temperature spike correlates to the remaining lifespan curve.)