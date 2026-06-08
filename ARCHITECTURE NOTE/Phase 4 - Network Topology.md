## Assumptions & Logic:

* We will implement PyTorch specially the "nn" Module, it stands for nueral network, PyTorch provides all the necessary building blocks, trainable layers, and loss functions needed to design deep learning models. It handles the background management of weights, biases, and structural GPU acceleration so we do not have to write them manually. (Search for it to know more, i suggest reading about the difference between it and Tensorflow)

* We will create a custom neural network Class, This class will inherit from nn.Module which is the foundational base class for all neural network architectures in PyTorch.

* We will stick to a kernel size of 3 that slides throughout each of the 30 cycle, And for the out_channel we will do 32, it's standard here, the limited number of sensors produce limited amount of failuire patterns no need to go crazy. increasing this leads to the model being "too smart" it will start to memorize useless patterns thinking that they're actual failuires.

* 64 neurons for our linear functions is sufficient as well it's a balance between computational complexity and the brain power needed to extract relationships between the features.

* Forward function is mandatory here since it declares the order of execution, when we pass the data to our model instance PyTorch inetrcepts it and runs the Forward function to have an idea of what the correct order is.


## Pytorch nn used module dict:
* nn.Conv1d
* * 1-Dimensional Convolutional Layer. It slides a mathematical window across the cycles to convert raw, meaningless sensor numbers into geometric features (slopes, spikes, drops). It extracts narrative of what't happening with these sensor results.

* nn.ReLU
* * Rectified Linear Unit activation function. The noise destroyer. It mathematically forces any negative number to exactly 0. This instantly strips out useless signals

* nn.MaxPool1d
* * 1-Dimensional Max Pooling Layer. The compressor. It slides a window across the features and only keeps the highest number, discarding the rest.it cuts the computational friction in half and prevents the model from obsessing over the exact cycle an anomaly occurred.

* nn.Flatten
* * Flattens dimensions into a 1D tensor. Linear layers mathematically cannot process 3D blocks or 2D feature maps. Flatten crushes your extracted data into a single, flat 1D vector so the standard logic gates can read it.

* nn.Linear
* * The Function: Applies a linear transformation to the incoming data. Also known as the "dense" layer. This is where the network stops looking at how data/shapes changes overtime and starts weighing all the extracted clues against each other to form a final output.

* nn.Sigmoid
* * Applies the Sigmoid mathematical function. A Linear layer might output a raw number like 42.7 or -8.4. Sigmoid forcibly compresses any number into a strict range between 0.0 and 1.0. It translates raw math into an actionable, real-world percentage.

