## Assumptions & Logic:

* We implement PyTorch's nn.Module. PyTorch handles the foundational management of weights, biases, backpropagation gradients, and GPU acceleration.

* We utilize a kernel_size of 3 sliding across the 30-cycle window, outputting to 32 feature channels. This is an optimal constraint; increasing filter complexity risks the model memorizing simulation noise rather than genuine failure mechanics.

* The linear regression head utilizes 64 neurons—a balance between computational efficiency and the capacity needed to weigh extracted relationships.

* The forward() function dictates the strict order of mathematical execution when data passes through the network.


## Pytorch nn used module dict:
* nn.Conv1d
* * 1-Dimensional Convolutional Layer. It slides a mathematical window across the cycles to convert raw sensor numbers into geometric features (slopes, spikes, drops), extracting the degradation narrative.

* nn.ReLU
* * Rectified Linear Unit activation function. It mathematically forces any negative number to exactly 0, instantly stripping out useless signals.

* nn.MaxPool1d
* * 1-Dimensional Max Pooling Layer. It slides a window across the features and retains only the highest activation, halving computational friction and preventing the network from over-indexing on the exact cycle an anomaly occurred.

* nn.Flatten
* * Crushes the extracted 3D/2D feature maps into a single 1D vector so the standard linear logic gates can process them.

* nn.Linear
* * Applies a linear transformation to the incoming data. This "dense" layer weighs all extracted clues against each other to output the final, unbounded Remaining Useful Life (RUL) prediction.