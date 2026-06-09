## Assumptions & Logic:

* Convert the numpy arrays we got from Phase_3 to Tensorflow float32 since that what PyTorch requires

* We use Binary Cross Entropy (BCELoss) as our Loss Function, It's the standard for binary classification, The goal here is as for most loss functions is to minimize it to increase our model performance. Search for the Math behind it if interested.


* As an optimizer for the learning rate we use Adam (Adaptive Moment Estimation). it dynamically adjusts the "learning rate" (how aggressively it changes the weights) for every single CNN filter.

* Create the learning loop (Backpropagation):
These are the core operations of DL it's a 5 step process per batch:
forward(): Make a prediction.
loss(): Calculate the error.
zero_grad(): Clear out the old adjustments from the previous batch.
backward(): Calculate the exact mathematical adjustments needed to fix the error (Derivatives).
step(): Actually apply those adjustments to the network's weights.

* Some numbers:
* * We will do batches of 64 to be kind to our machines
* * the learning rate of 0.001 is standard here, increasing or decreasing this will tell Adam to either aggresively overcorrect the loss function or take alot of time to learn and update the weights
* * 15 Epochs is also a sweetspot, increasing or decreasing this results in the model 
overfitting(It has seen the data way too much and starting to memorize instead of predict)
underfitting(It has seen the data way too low to figuire out what a temp spike looks like)