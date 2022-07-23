# MLP - Neural Net
![neuron image](neuron.png)

While knowledge of the math behind a neural net would be preferred this isn't strictly needed to start a beginner Neural Net implementation as most modern high-level ML frameworks such as Pytorch provides functions helping us. It is however needed to have some understanding of the building blocks that makes up our network and what we wish to accomplish.

The point in our case is to learn a mapping $f$ from input vector $\textbf{x}$ to correct output $y$ s.t. $f(\textbf{x}) = y$. The header image shows one fully connected `neuron`, as an example of a mapping. It is connected to all inputs, multiplies inputs by a weight $w$ and adds a scalar bias $b$. These are trainable parameters modified during training by a gradient to better fit our mapping. Finally, this is sent through an activation function to generate an output.

We can stack several neurons together in a layer, and then add several layers to make the layers learn different properties of our data and thus a more complex mapping. This structure is called network model or architecture and an example image is shown below.
![model image](neural_net.png)

Layers, network images

Sending the input from left to right is c`forward pass`


One often uses specialized weight layers depending on the input data. Common ones are convolutions for computer vision, recurrency for timeseries and linear (fully connected) layers for numerical input.

init weights
forward pass, 

## Activation function


prediction,
## Loss Function

## Optimizer
Adam
backward propagation, weights
