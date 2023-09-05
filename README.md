# Neural Networks in R

## Description
Implementations and anlysis of 1-layer artificial neural networks in R, aimed at predicting the species of a bird based on its wing length and body weight. Softmax activation function also implemented and used on the output layer to handle the one-hot encoded response. Tanh and ReLU activations are both tested on the hidden layer, with validation analyses being done for each. Cross-entopy loss function used for training, with the 'nlm' optimizer in R being used to perform the parameter optimization. L1 regularisation is also implemented and tested, with the optimal nu (L1 penalty multiplier) value being found using the validation sets.

The dataset, Hawks_Data_2023.txt, contains 148 observations with the following variables:
- SpecA, SpecB, SpecC: One-hot encoded variables indicating the species (A, B, or C).
- Wing: Length in meters of the wing.
- Weight: Body weight in kilograms.
